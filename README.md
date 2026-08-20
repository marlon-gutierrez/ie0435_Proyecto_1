# Contaminated Sample Detection Using Digital Signal Processing and Machine Learning

This repository contains the development of **Project 1 for IE0435 - Artificial Intelligence Applied to Electrical Engineering** at the University of Costa Rica.

The project implements a complete **digital signal processing and supervised machine learning pipeline** for the automatic detection of contaminated samples.

The system processes image-based signals, performs segmentation and feature extraction, and evaluates multiple machine learning algorithms to determine whether a sample belongs to the positive or negative class.

---

## Project Overview

The complete processing pipeline can be summarized as:

**RGB Signal → Intensity Representation → Kittler Thresholding → Binary Signal → Feature Extraction → Machine Learning Classification**

The project uses two independent datasets:

- **120 training samples**
  - 60 positive samples
  - 60 negative samples
- **30 independent test samples**

The same preprocessing and feature extraction pipeline is applied to both datasets.

---

## Signal Processing Pipeline

### 1. RGB Signal Acquisition

The original samples are represented as RGB images containing three signal components:

- Red
- Green
- Blue

The samples were acquired under similar lighting conditions to reduce unwanted variations in intensity caused by illumination changes, shadows, and reflections.

---

### 2. Intensity Signal Generation

Each RGB sample is transformed from its three original components into a single intensity representation.

This reduces every pixel from three values to one intensity value representing its brightness.

The resulting intensity signal is then used as the input for the segmentation stage.

---

### 3. Signal Segmentation

The intensity signals are segmented using the **Kittler Minimum Error Thresholding Algorithm**.

Instead of applying a fixed threshold to the complete dataset, the threshold is calculated independently for every sample.

This allows the preprocessing stage to adapt to small variations in signal intensity between samples.

---

### 4. Binary Signal Generation

After thresholding, each sample is represented as a binary signal.

The processed samples are resized to:

```text
128 × 128
```

This produces a standardized representation that can be used during dataset generation and feature extraction.

The resulting binary samples are also inspected to verify that the segmentation process correctly separates the relevant elements from the background.

---

## Dataset Generation

Each processed sample is represented as a `128 × 128` matrix containing:

```text
16,384 values
```

The matrix is flattened into a single row vector and stored in a CSV file.

Therefore, each row in the dataset represents one complete processed sample.

The final column contains the corresponding class label:

- `1` → Positive sample
- `0` → Negative sample

---

## Dataset Verification

To verify the integrity of the generated CSV files, the preprocessing program reconstructs each binary sample directly from its corresponding row vector.

The reconstructed samples are stored separately and compared with the original processed samples.

This procedure verifies that the transformation:

```text
Binary Signal → Row Vector → CSV → Reconstructed Signal
```

preserves the original information correctly.

---

## Feature Extraction

Instead of directly using all **16,384 values** as inputs to the machine learning algorithms, the system extracts **17 representative features** from each processed sample.

These features describe geometric, statistical, morphological, and population properties of the detected elements.

The extracted features are:

1. Number of detected objects
2. Mean object area
3. Area standard deviation
4. Mean aspect ratio
5. Aspect ratio standard deviation
6. Mean solidity
7. Solidity standard deviation
8. Mean elongation
9. Elongation standard deviation
10. Mean compactness
11. Number of contamination candidates
12. Contamination fraction
13. Area coefficient of variation
14. Aspect ratio coefficient of variation
15. Number of objects separated using watershed
16. Population score
17. Area entropy

This reduces the input representation from:

```text
16,384 values → 17 extracted features
```

The resulting feature vector provides a much more compact representation of each sample for the classification stage.

---

## Separation of Overlapping Objects

Some detected elements may appear connected or overlapping after segmentation.

To improve the feature extraction process, the system implements the **Watershed algorithm**.

A distance transform is used to estimate individual object centers and separate connected regions.

This allows the system to obtain more representative geometric and population-based features from each sample.

---

## Data Normalization

The extracted features have different numerical ranges and scales.

For this reason, feature normalization is applied before training the classification algorithms.

This preprocessing stage improves the stability of algorithms that are sensitive to differences in feature magnitude, particularly Support Vector Machines and K-Nearest Neighbors.

---

## Machine Learning Models

Several supervised machine learning algorithms were trained and evaluated.

### Decision Tree

A classification model based on hierarchical decision rules applied to the extracted features.

### Gaussian Naive Bayes

A probabilistic classifier based on Bayes' theorem and Gaussian feature distributions.

### Support Vector Machine - RBF Kernel

Three different values of the regularization parameter `C` were evaluated:

```text
C = 1
C = 10
C = 100
```

### Linear Support Vector Machine

A Support Vector Machine using a linear decision boundary.

### K-Nearest Neighbors

Two configurations were evaluated:

```text
k = 3
k = 5
```

All models were trained using the training dataset and evaluated using the independent test dataset.

---

## Evaluation Metrics

The performance of each classifier was evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

The final model was selected primarily according to the highest **F1 Score**, providing a balance between Precision and Recall.

---

## Best Model

The best-performing classifier was:

### SVM with RBF Kernel — C = 1

| Metric | Result |
| --- | ---: |
| Accuracy | **0.9000** |
| Precision | **0.8750** |
| Recall | **0.9333** |
| F1 Score | **0.9032** |

The SVM with an RBF kernel and `C = 1` obtained the highest F1 Score among the evaluated classifiers and showed the best overall performance on the independent test dataset.

The trained model is stored as:

```text
C33619_Marlon_Gutierrez.joblib
```

---

## Repository Structure

```text
ie0435_Proyecto_1/
│
├── conjunto_entrenamiento/
│   ├── CSV/
│   ├── imagenes_RGB/
│   ├── imagenes_binarias/
│   └── imagenes_reconstruidas/
│
├── conjunto_prueba/
│   ├── CSV/
│   ├── imagenes_RGB/
│   ├── imagenes_binarias/
│   └── imagenes_reconstruidas/
│
├── mejor_modelo/
│   └── C33619_Marlon_Gutierrez.joblib
│
├── src/
│   ├── preprocesamiento_digital_imagenes.py
│   └── entrenamiento_y_prueba.py
│
└── README.md
```

---

## Main Programs

### `preprocesamiento_digital_imagenes.py`

Responsible for:

- Loading RGB samples
- Computing intensity representations
- Applying Kittler thresholding
- Generating binary signals
- Resizing samples to `128 × 128`
- Generating CSV datasets
- Reconstructing samples from CSV data
- Verifying dataset integrity

### `entrenamiento_y_prueba.py`

Responsible for:

- Loading the generated datasets
- Extracting the 17 representative features
- Separating overlapping objects using Watershed
- Normalizing the extracted features
- Training multiple supervised machine learning models
- Evaluating classifier performance
- Comparing Accuracy, Precision, Recall, and F1 Score
- Selecting the best-performing model
- Saving the final trained classifier

---

## Technologies and Methods

- Python
- OpenCV
- Digital Signal Processing
- Signal Segmentation
- Kittler Thresholding
- Binary Signal Processing
- Feature Extraction
- Watershed Segmentation
- Data Normalization
- Supervised Machine Learning
- Support Vector Machines
- Decision Trees
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Classification Metrics

---

## Course

**IE0435 - Artificial Intelligence Applied to Electrical Engineering**  
School of Electrical Engineering  
University of Costa Rica  
I Semester, 2026

---

## Author

**Marlon Gutiérrez Vásquez**  
Electrical Engineering  
University of Costa Rica
