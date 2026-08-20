import os
import joblib
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================

# Directorio principal del proyecto. A partir de esta ruta se construyen las demás rutas.
# Coloque la direccion que corresponda a la direccion donde guardó el repositorio
BASE_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Archivos CSV de entrenamiento y prueba.
RUTA_TRAIN = os.path.join(BASE_PROYECTO, "conjunto_entrenamiento", "CSV", "CSV.csv")
RUTA_TEST  = os.path.join(BASE_PROYECTO, "conjunto_prueba",        "CSV", "CSV.csv")

# Carpeta y nombre del archivo donde se guarda el mejor modelo entrenado.
RUTA_SALIDA = os.path.join(BASE_PROYECTO, "mejor_modelo")
NOMBRE_MODELO = "C33619_Marlon_Gutierrez.joblib"
RUTA_MODELO_FINAL = os.path.join(RUTA_SALIDA, NOMBRE_MODELO)

# Cantidades esperadas en los CSV y tamaño de cada imagen binaria.
N_DATOS_ENTRENAMIENTO_ESPERADOS = 120
N_DATOS_PRUEBA_ESPERADOS        = 30
N_PIXELES_ESPERADOS  = 128 * 128
N_COLUMNAS_ESPERADAS = N_PIXELES_ESPERADOS + 1
FORMA_IMAGEN = (128, 128)


# =============================================================================
# SEPARACIÓN DE OBJETOS SUPERPUESTOS (WATERSHED)
# =============================================================================

def separar_superpuestos(img_bin):
    """
    Aplica watershed para intentar separar objetos que aparecen pegados.
    Retorna los contornos separados y la cantidad de contornos encontrados.
    """
    # Se calcula el mapa de distancias para ubicar zonas centrales de los objetos.
    dist = cv2.distanceTransform(img_bin, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # Se obtienen regiones seguras de primer plano que sirven como semillas.
    dist_uint8 = (dist_norm * 255).astype(np.uint8)
    _, sure_fg = cv2.threshold(dist_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Se define la zona desconocida entre fondo y primer plano.
    kernel   = np.ones((3, 3), np.uint8)
    sure_bg  = cv2.dilate(img_bin, kernel, iterations=2)
    unknown  = cv2.subtract(sure_bg, sure_fg)

    # Se crean las etiquetas iniciales que usa el algoritmo watershed.
    _, markers = cv2.connectedComponents(sure_fg)
    markers   = markers + 1
    markers[unknown == 255] = 0

    # Watershed requiere una imagen en formato BGR.
    img_bgr = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_bgr, markers)

    # Se reconstruye una máscara con los objetos separados.
    mask_sep = np.zeros_like(img_bin)
    mask_sep[markers > 1] = 255

    contornos, _ = cv2.findContours(mask_sep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos, len(contornos)


# =============================================================================
# EXTRACCIÓN ADAPTATIVA DE CARACTERÍSTICAS
# =============================================================================

def extraer_caracteristicas_imagen(pixels_fila):
    """
    Convierte una fila del CSV en imagen y extrae 17 características geométricas
    y poblacionales para usarlas como entrada de los modelos de clasificación.
    """

    # Reconstrucción de la imagen de 128x128 a partir del vector de pixeles.
    img = (pixels_fila.reshape(FORMA_IMAGEN) * 255).astype(np.uint8)
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # Detección de contornos iniciales, eliminando regiones muy pequeñas.
    contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [c for c in contornos if cv2.contourArea(c) >= 5]

    if not contornos:
        return np.zeros(17)

    # Cálculo de áreas para definir umbrales adaptativos según la imagen.
    areas_todas = np.array([cv2.contourArea(c) for c in contornos])
    p10 = np.percentile(areas_todas, 10)
    p90 = np.percentile(areas_todas, 90)
    area_mediana = np.median(areas_todas)

    area_min_adaptativa = max(5,   area_mediana * 0.05)
    area_max_adaptativa = min(5000, area_mediana * 5.0)

    # Rango de aspecto usado como criterio para identificar objetos tipo arroz.
    aspecto_min = 1.5
    aspecto_max = 5.0

    # Si un contorno tiene baja solidez, se intenta separar con watershed.
    contornos_watershed, n_extra_watershed = [], 0
    for c in contornos:
        area = cv2.contourArea(c)
        if area < 5:
            continue
        casco = cv2.convexHull(c)
        area_c = cv2.contourArea(casco)
        solidez = area / area_c if area_c > 0 else 1.0
        if solidez < 0.65:
            mask_local = np.zeros_like(img)
            cv2.drawContours(mask_local, [c], -1, 255, -1)
            cnts_sep, n_sep = separar_superpuestos(mask_local)
            if n_sep > 1:
                contornos_watershed.extend(cnts_sep)
                n_extra_watershed += n_sep - 1
            else:
                contornos_watershed.append(c)
        else:
            contornos_watershed.append(c)

    contornos_finales = contornos_watershed if contornos_watershed else contornos
    n_objetos = len(contornos_finales)

    # Listas donde se almacenan las mediciones calculadas para cada objeto.
    areas        = []
    aspectos     = []
    solideces    = []
    elongaciones = []
    compacidades = []
    n_arroz      = 0

    for c in contornos_finales:
        area = cv2.contourArea(c)
        if area < 5:
            continue

        areas.append(area)

        _, _, w, h = cv2.boundingRect(c)
        lado_mayor = max(w, h, 1)
        lado_menor = min(w, h, 1)
        aspecto = lado_mayor / lado_menor
        aspectos.append(aspecto)

        casco = cv2.convexHull(c)
        area_casco = cv2.contourArea(casco)
        solidez = area / area_casco if area_casco > 0 else 0
        solideces.append(solidez)

        if len(c) >= 5:
            (_, _), (eje_min, eje_may), _ = cv2.fitEllipse(c)
            eje_may = max(eje_may, 1)
            elongacion = 1.0 - (eje_min / eje_may)
        else:
            elongacion = 0.0
        elongaciones.append(elongacion)

        perimetro  = cv2.arcLength(c, True)
        compacidad = (4 * np.pi * area / (perimetro ** 2)) if perimetro > 0 else 0
        compacidades.append(compacidad)

        # Se cuenta como arroz si cumple criterios de solidez, aspecto y área.
        es_arroz = (
            solidez    >  0.78 and
            aspecto_min <= aspecto <= aspecto_max and
            area_min_adaptativa <= area <= area_max_adaptativa
        )
        if es_arroz:
            n_arroz += 1

    def media(lst): return float(np.mean(lst)) if lst else 0.0
    def std(lst):   return float(np.std(lst))  if lst else 0.0

    fraccion_arroz = n_arroz / max(n_objetos, 1)

    # Características poblacionales que resumen variación y similitud entre objetos.
    if areas:
        cv_area   = std(areas)   / (media(areas)   + 1e-6)
        cv_aspecto = std(aspectos) / (media(aspectos) + 1e-6)
    else:
        cv_area = cv_aspecto = 0.0

    if areas:
        mediana_area  = float(np.median(areas))
        dentro_rango  = sum(1 for a in areas if 0.5 * mediana_area <= a <= 1.5 * mediana_area)
        score_pob     = dentro_rango / len(areas)
    else:
        score_pob = 0.0

    if len(areas) > 1:
        hist, _ = np.histogram(areas, bins=8)
        hist    = hist / (hist.sum() + 1e-6)
        entropia_area = float(-np.sum(hist * np.log2(hist + 1e-9)))
    else:
        entropia_area = 0.0

    # Vector final de 17 características que representa la imagen.
    return np.array([
        n_objetos,
        media(areas),      std(areas),
        media(aspectos),   std(aspectos),
        media(solideces),  std(solideces),
        media(elongaciones), std(elongaciones),
        media(compacidades),
        n_arroz,
        fraccion_arroz,
        cv_area,
        cv_aspecto,
        float(n_extra_watershed),
        score_pob,
        entropia_area,
    ])


def extraer_caracteristicas(X, nombre="conjunto"):
    # Aplica la extracción de características a todas las imágenes del conjunto.
    print(f"\nExtrayendo características de {nombre}...")
    features = np.array([extraer_caracteristicas_imagen(fila) for fila in X])
    print(f"Vector de características: {features.shape[1]} por imagen")
    return features


# =============================================================================
# CARGA DE CSV
# =============================================================================

def cargar_csv(ruta_csv, nombre_conjunto, n_datos_esperados):
    # Carga el CSV, verifica tamaño esperado y separa pixeles de etiquetas.
    print("\n" + "=" * 70)
    print(f"CARGANDO {nombre_conjunto.upper()}")
    print("=" * 70)

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el CSV:\n{ruta_csv}")

    df = pd.read_csv(ruta_csv, header=0)
    if df.shape[0] != n_datos_esperados:
        print(f"ADVERTENCIA: {n_datos_esperados} filas esperadas, {df.shape[0]} encontradas.")
    if df.shape[1] != N_COLUMNAS_ESPERADAS:
        raise ValueError(f"Se esperaban {N_COLUMNAS_ESPERADAS} columnas; hay {df.shape[1]}.")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    print(f"Muestras: {X.shape[0]}  |  Negativas: {np.sum(y==0)}  |  Positivas: {np.sum(y==1)}")
    return X, y


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

def preprocesar_datos(X_train_feat, X_test_feat):
    # Estandariza las características para que tengan escala comparable.
    scaler = StandardScaler()
    X_train_esc = scaler.fit_transform(X_train_feat)
    X_test_esc  = scaler.transform(X_test_feat)
    return X_train_esc, X_test_esc, scaler


# =============================================================================
# MODELOS
# =============================================================================

# Modelos que se entrenan y comparan usando el mismo conjunto de características.
MODELOS = {
    "Arbol de Decision": DecisionTreeClassifier(random_state=42),
    "Naive Bayes":       GaussianNB(),
    "SVM RBF C1":        SVC(kernel="rbf", C=1.0,   gamma="scale", random_state=42),
    "SVM RBF C10":       SVC(kernel="rbf", C=10.0,  gamma="scale", random_state=42),
    "SVM RBF C100":      SVC(kernel="rbf", C=100.0, gamma="scale", random_state=42),
    "SVM Lineal":        SVC(kernel="linear", C=1.0, random_state=42),
    "KNN k=3":           KNeighborsClassifier(n_neighbors=3),
    "KNN k=5":           KNeighborsClassifier(n_neighbors=5),
}


# =============================================================================
# EVALUACIÓN
# =============================================================================

def evaluar_modelo(modelo, X_test, y_test):
    # Calcula predicciones, métricas y matriz de confusión del modelo evaluado.
    y_pred = modelo.predict(X_test)
    metricas = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
    }
    matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return metricas, matriz, y_pred


def imprimir_matriz_confusion(matriz):
    # Muestra la matriz de confusión en un formato más fácil de leer en terminal.
    print("\nMatriz de confusión:")
    print("                 Predicción")
    print("              Limpia  Contaminada")
    print(f"Real limpia        {matriz[0,0]:2d}        {matriz[0,1]:2d}")
    print(f"Real contaminada   {matriz[1,0]:2d}        {matriz[1,1]:2d}")
    print(f"\nVN={matriz[0,0]}  FP={matriz[0,1]}  FN={matriz[1,0]}  VP={matriz[1,1]}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO MEJORADO - DETECCIÓN DE GRANOS DE ARROZ")
    print("17 características: geométricas + adaptativas + poblacionales")
    print("=" * 70)

    # Carga de datos desde los CSV de entrenamiento y prueba.
    X_train_px, y_train = cargar_csv(RUTA_TRAIN, "entrenamiento", N_DATOS_ENTRENAMIENTO_ESPERADOS)
    X_test_px,  y_test  = cargar_csv(RUTA_TEST,  "prueba",        N_DATOS_PRUEBA_ESPERADOS)

    # Transformación de pixeles a características numéricas.
    X_train_feat = extraer_caracteristicas(X_train_px, "entrenamiento")
    X_test_feat  = extraer_caracteristicas(X_test_px,  "prueba")

    X_train, X_test, scaler = preprocesar_datos(X_train_feat, X_test_feat)

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO Y EVALUACIÓN")
    print("=" * 70)

    resultados = {}
    modelos_entrenados   = {}
    predicciones_guardadas = {}

    # Entrena cada modelo, lo evalúa y guarda sus resultados.
    for nombre_modelo, modelo in MODELOS.items():
        print(f"\nEntrenando: {nombre_modelo}")
        modelo.fit(X_train, y_train)
        metricas, matriz, y_pred = evaluar_modelo(modelo, X_test, y_test)
        resultados[nombre_modelo]             = metricas
        modelos_entrenados[nombre_modelo]     = modelo
        predicciones_guardadas[nombre_modelo] = y_pred
        print(f"Accuracy={metricas['Accuracy']:.4f}  Precision={metricas['Precision']:.4f}"
              f"  Recall={metricas['Recall']:.4f}  F1={metricas['F1']:.4f}")
        imprimir_matriz_confusion(matriz)

    print("\n" + "=" * 70)
    print("TABLA COMPARATIVA")
    print("=" * 70)
    df_res = pd.DataFrame(resultados).T[["Accuracy", "Precision", "Recall", "F1"]].round(4)
    print(df_res.to_string())

    # Selección del mejor modelo según el valor de F1 Score.
    mejor_nombre = df_res["F1"].idxmax()
    mejor_modelo = modelos_entrenados[mejor_nombre]
    mejores_metricas = df_res.loc[mejor_nombre].to_dict()
    mejores_pred     = predicciones_guardadas[mejor_nombre]

    print("\n" + "=" * 70)
    print("MEJOR MODELO")
    print("=" * 70)
    print(f"Modelo: {mejor_nombre}  |  F1={mejores_metricas['F1']:.4f}")

    # Impresión de las predicciones del mejor modelo para cada muestra de prueba.
    for i, (real, pred) in enumerate(zip(y_test, mejores_pred), 1):
        estado = "CORRECTA" if real == pred else "INCORRECTA"
        print(f"  #{i:02d}: real={real}  pred={pred}  → {estado}")

    # Se guarda el modelo junto con el escalador y la información necesaria para reutilizarlo.
    os.makedirs(RUTA_SALIDA, exist_ok=True)
    paquete = {
        "modelo":        mejor_modelo,
        "scaler":        scaler,
        "nombre_modelo": mejor_nombre,
        "metricas":      mejores_metricas,
        "clases":        {0: "negativa", 1: "positiva (arroz)"},
        "n_pixeles":     N_PIXELES_ESPERADOS,
        "forma_imagen":  FORMA_IMAGEN,
        "pipeline":      "17 características adaptativas + StandardScaler",
    }
    joblib.dump(paquete, RUTA_MODELO_FINAL)
    print(f"\nModelo guardado en: {RUTA_MODELO_FINAL}")


if __name__ == "__main__":
    main()
