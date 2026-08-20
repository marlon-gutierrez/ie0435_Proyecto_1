import cv2
import os
import numpy as np
import csv


# Ruta principal del proyecto. A partir de esta carpeta se construyen las rutas
# para entrenamiento, prueba, CSV e imágenes reconstruidas.
# Coloque la direccion que corresponda a la direccion donde guardó el repositorio

directorio_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# =========================================================
# DIRECTORIOS DEL CONJUNTO DE ENTRENAMIENTO
# =========================================================

# Carpeta donde se encuentran las imágenes RGB originales de entrenamiento.
directorio_rgb_entrenamiento = os.path.join(
    directorio_base,
    r"conjunto_entrenamiento\imagenes_RGB"
)

directorio_rgb_entrenamiento = os.path.join(
    directorio_base,
    "conjunto_entrenamiento", "imagenes_RGB"
)

directorio_segmentadas_entrenamiento = os.path.join(
    directorio_base,
    "conjunto_entrenamiento", "imagenes_binarias"
)

directorio_csv_entrenamiento = os.path.join(
    directorio_base,
    "conjunto_entrenamiento", "CSV"
)

directorio_reconstruidas_entrenamiento = os.path.join(
    directorio_base,
    "conjunto_entrenamiento", "imagenes_reconstruidas"
)

ruta_csv_entrenamiento = os.path.join(
    directorio_csv_entrenamiento,
    "CSV.csv"
)

directorio_rgb_prueba = os.path.join(
    directorio_base,
    "conjunto_prueba", "imagenes_RGB"
)

directorio_segmentadas_prueba = os.path.join(
    directorio_base,
    "conjunto_prueba", "imagenes_binarias"
)

directorio_csv_prueba = os.path.join(
    directorio_base,
    "conjunto_prueba", "CSV"
)

directorio_reconstruidas_prueba = os.path.join(
    directorio_base,
    "conjunto_prueba", "imagenes_reconstruidas"
)

ruta_csv_prueba = os.path.join(
    directorio_csv_prueba,
    "CSV.csv"
)

# Crear las carpetas de salida en caso de que todavía no existan.
os.makedirs(directorio_segmentadas_entrenamiento, exist_ok=True)
os.makedirs(directorio_csv_entrenamiento, exist_ok=True)
os.makedirs(directorio_reconstruidas_entrenamiento, exist_ok=True)

os.makedirs(directorio_segmentadas_prueba, exist_ok=True)
os.makedirs(directorio_csv_prueba, exist_ok=True)
os.makedirs(directorio_reconstruidas_prueba, exist_ok=True)


# Dimensiones finales usadas para todas las imágenes antes de vectorizarlas.
ancho = 128
alto = 128


def calcular_umbral_kittler(imagen_intensidad):
    """Calcula el umbral de segmentación usando el método de Kittler."""

    # Se calcula el histograma de intensidades de la imagen en escala de grises.
    histograma = cv2.calcHist(
        [imagen_intensidad],
        [0],
        None,
        [256],
        [0, 256]
    )

    histograma = histograma.ravel().astype(np.float64)

    suma_histograma = np.sum(histograma)

    if suma_histograma == 0:
        return 0

    # Se normaliza el histograma para trabajar con probabilidades.
    histograma = histograma / suma_histograma

    niveles = np.arange(256)

    mejor_umbral = 0
    menor_criterio = np.inf

    # Se prueba cada posible umbral y se elige el que minimiza el criterio.
    for umbral in range(1, 255):

        clase_1 = histograma[:umbral + 1]
        niveles_1 = niveles[:umbral + 1]

        clase_2 = histograma[umbral + 1:]
        niveles_2 = niveles[umbral + 1:]

        w1 = np.sum(clase_1)
        w2 = np.sum(clase_2)

        if w1 <= 0 or w2 <= 0:
            continue

        media_1 = np.sum(niveles_1 * clase_1) / w1
        media_2 = np.sum(niveles_2 * clase_2) / w2

        varianza_1 = np.sum(((niveles_1 - media_1) ** 2) * clase_1) / w1
        varianza_2 = np.sum(((niveles_2 - media_2) ** 2) * clase_2) / w2

        if varianza_1 <= 0 or varianza_2 <= 0:
            continue

        criterio = (
            1
            + 2 * (
                w1 * np.log(np.sqrt(varianza_1))
                + w2 * np.log(np.sqrt(varianza_2))
            )
            - 2 * (
                w1 * np.log(w1)
                + w2 * np.log(w2)
            )
        )

        if criterio < menor_criterio:
            menor_criterio = criterio
            mejor_umbral = umbral

    # Ajuste para imágenes casi completamente blancas.
    if mejor_umbral > 250:
        mejor_umbral = 150

    return mejor_umbral


def limpiar_directorio_imagenes(directorio):
    """Elimina imágenes previas de un directorio para evitar archivos duplicados."""

    for archivo in os.listdir(directorio):

        if archivo.lower().endswith((".png", ".jpg", ".jpeg")):

            ruta_archivo = os.path.join(
                directorio,
                archivo
            )

            os.remove(ruta_archivo)


def cargar_imagen_intensidad(directorio_rgb, nombre_base):
    """Busca una imagen por nombre, la carga, la redimensiona y la pasa a gris."""

    posibles_extensiones = [
        ".png",
        ".PNG",
        ".jpeg",
        ".JPEG",
        ".jpg",
        ".JPG"
    ]

    ruta_imagen = None

    # Se revisan varias extensiones para aceptar imágenes PNG, JPG o JPEG.
    for extension in posibles_extensiones:

        ruta_temporal = os.path.join(
            directorio_rgb,
            nombre_base + extension
        )

        if os.path.exists(ruta_temporal):
            ruta_imagen = ruta_temporal
            break

    if ruta_imagen is None:
        print(f"No se encontro la imagen: {nombre_base}")
        return None

    imagen_rgb = cv2.imread(ruta_imagen)

    if imagen_rgb is None:
        print(f"Error al cargar la imagen: {nombre_base}")
        return None

    # Todas las imágenes se llevan al mismo tamaño para que el CSV tenga
    # siempre la misma cantidad de columnas.
    imagen_rgb = cv2.resize(
        imagen_rgb,
        (ancho, alto),
        interpolation=cv2.INTER_AREA
    )

    imagen_intensidad = cv2.cvtColor(
        imagen_rgb,
        cv2.COLOR_BGR2GRAY
    )

    return imagen_intensidad


def segmentar_y_vectorizar_imagen(
    directorio_rgb,
    directorio_segmentadas,
    nombre_base,
    etiqueta
):
    """Segmenta una imagen, la guarda y devuelve su vector de pixeles con etiqueta."""

    imagen_intensidad = cargar_imagen_intensidad(
        directorio_rgb,
        nombre_base
    )

    if imagen_intensidad is None:
        return None

    umbral_kittler = calcular_umbral_kittler(
        imagen_intensidad
    )

    # La imagen en escala de grises se convierte a una imagen binaria 0/255.
    _, imagen_segmentada_255 = cv2.threshold(
        imagen_intensidad,
        umbral_kittler,
        255,
        cv2.THRESH_BINARY
    )

    # Guardar imagen segmentada solamente en formato PNG.
    ruta_segmentada = os.path.join(
        directorio_segmentadas,
        nombre_base + ".png"
    )

    cv2.imwrite(
        ruta_segmentada,
        imagen_segmentada_255
    )

    # Convertir imagen de 0/255 a 0/1 para guardarla en el CSV.
    imagen_segmentada_01 = (
        imagen_segmentada_255 / 255
    ).astype(np.uint8)

    # La matriz de pixeles se convierte en una sola fila.
    vector_fila = imagen_segmentada_01.flatten()

    # Al final de la fila se agrega la etiqueta de clase.
    vector_fila_con_etiqueta = np.append(
        vector_fila,
        etiqueta
    )

    vector_fila_con_etiqueta = vector_fila_con_etiqueta.astype(int)

    print(f"Imagen procesada correctamente: {nombre_base}")
    print(f"Guardada como: {nombre_base}.png")
    print(f"Umbral Kittler utilizado: {umbral_kittler}")
    print(f"Etiqueta agregada en la ultima columna: {etiqueta}\n")

    return vector_fila_con_etiqueta


def procesar_dataset(
    nombre_dataset,
    imagenes,
    directorio_rgb,
    directorio_segmentadas,
    ruta_csv,
    directorio_reconstruidas
):
    """Procesa un conjunto completo: segmentación, CSV y reconstrucción."""

    print("\n===================================")
    print(f"PROCESANDO CONJUNTO {nombre_dataset}")
    print("===================================\n")

    # Limpiar imágenes anteriores para evitar duplicados en PNG/JPG.
    limpiar_directorio_imagenes(directorio_segmentadas)
    limpiar_directorio_imagenes(directorio_reconstruidas)

    # Crear encabezado del CSV: pixel_1, pixel_2, ..., pixel_16384, etiqueta.
    encabezado = []

    for i in range(ancho * alto):
        encabezado.append(f"pixel_{i + 1}")

    encabezado.append("etiqueta")

    # Se segmenta cada imagen y se escribe una fila por imagen en el CSV.
    with open(ruta_csv, mode="w", newline="") as archivo_csv:

        escritor = csv.writer(archivo_csv)

        escritor.writerow(encabezado)

        for nombre_imagen, etiqueta in imagenes:

            fila = segmentar_y_vectorizar_imagen(
                directorio_rgb,
                directorio_segmentadas,
                nombre_imagen,
                etiqueta
            )

            if fila is not None:
                escritor.writerow(fila)

    # Reconstruir imágenes desde el CSV para verificar que los datos se guardaron correctamente.
    with open(ruta_csv, mode="r", newline="") as archivo_csv:

        lector = csv.reader(archivo_csv)

        next(lector)

        for numero_fila, fila in enumerate(lector, start=1):

            fila = np.array(
                fila,
                dtype=np.uint8
            )

            pixeles = fila[:-1]
            etiqueta = fila[-1]

            imagen_reconstruida = (
                pixeles.reshape(alto, ancho) * 255
            ).astype(np.uint8)

            # Para entrenamiento se conserva positiva/negativa.
            if nombre_dataset == "ENTRENAMIENTO":

                if etiqueta == 1:
                    nombre_reconstruida = f"positiva{numero_fila}.png"
                else:
                    nombre_reconstruida = f"negativa{numero_fila - 60}.png"

            # Para prueba se conserva prueba1, prueba2, ..., prueba30.
            else:

                nombre_reconstruida = f"prueba{numero_fila}.png"

            ruta_salida = os.path.join(
                directorio_reconstruidas,
                nombre_reconstruida
            )

            cv2.imwrite(
                ruta_salida,
                imagen_reconstruida
            )

    print("\nProceso finalizado.")
    print(f"Imagenes segmentadas guardadas en: {directorio_segmentadas}")
    print(f"Imagenes reconstruidas guardadas en: {directorio_reconstruidas}")
    print(f"Archivo CSV generado en: {ruta_csv}")


# =========================================================
# IMAGENES DE ENTRENAMIENTO
# =========================================================

imagenes_entrenamiento = []

# Se agregan primero las 60 imágenes positivas con etiqueta 1.
for i in range(1, 61):
    imagenes_entrenamiento.append(
        (f"positiva{i}", 1)
    )

# Luego se agregan las 60 imágenes negativas con etiqueta 0.
for i in range(1, 61):
    imagenes_entrenamiento.append(
        (f"negativa{i}", 0)
    )


# =========================================================
# IMAGENES DE PRUEBA
# =========================================================

imagenes_prueba = []

# prueba1 a prueba15 son negativas.
for i in range(1, 16):
    imagenes_prueba.append(
        (f"prueba{i}", 0)
    )

# prueba16 a prueba30 son positivas.
for i in range(16, 31):
    imagenes_prueba.append(
        (f"prueba{i}", 1)
    )


# =========================================================
# EJECUCION
# =========================================================

# Se procesa primero el conjunto de entrenamiento.
procesar_dataset(
    "ENTRENAMIENTO",
    imagenes_entrenamiento,
    directorio_rgb_entrenamiento,
    directorio_segmentadas_entrenamiento,
    ruta_csv_entrenamiento,
    directorio_reconstruidas_entrenamiento
)

# Después se procesa el conjunto de prueba con el mismo procedimiento.
procesar_dataset(
    "PRUEBA",
    imagenes_prueba,
    directorio_rgb_prueba,
    directorio_segmentadas_prueba,
    ruta_csv_prueba,
    directorio_reconstruidas_prueba
)
