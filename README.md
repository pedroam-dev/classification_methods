# Classification Methods Project

Este proyecto implementa y compara diferentes algoritmos de clasificación utilizando dos datasets: **Breast Cancer** y **Student Performance**. Cada algoritmo está optimizado con validación cruzada y búsqueda de hiperparámetros.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Algoritmos Implementados](#algoritmos-implementados)
- [Datasets](#datasets)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Resultados](#resultados)
- [Características Técnicas](#características-técnicas)
- [Contribución](#contribución)

## Descripción

Este proyecto presenta una implementación completa de múltiples algoritmos de clasificación supervisada, enfocándose en:

- **Optimización de hiperparámetros** mediante GridSearchCV
- **Validación cruzada estratificada** para evaluación robusta
- **Visualizaciones comprensivas** de resultados
- **Comparación sistemática** entre algoritmos
- **Análisis de ensemble methods** vs clasificadores individuales

## Algoritmos implementados

### 1. Perceptrón Multicapa (MLP)
- **Archivos**: `mlp_breast_cancer.py`, `mlp_student_performance.py`
- **Características**:
  - Arquitecturas de red optimizadas (1-3 capas ocultas)
  - Multiple activation functions (ReLU, Tanh)
  - Solvers adaptativos (Adam, L-BFGS)
  - Early stopping para evitar overfitting
  - Análisis de curvas de aprendizaje

### 2. Voting Classifier
- **Archivos**: `vc_breast_cancer.py`, `vc_student_performance.py`
- **Características**:
  - Ensemble de 6 clasificadores diversos
  - Votación hard y soft
  - Comparación con clasificadores individuales
  - Análisis de mejora del ensemble

**Clasificadores incluidos**:
- Random Forest
- Support Vector Machine
- Logistic Regression
- Gradient Boosting
- Naive Bayes
- K-Nearest Neighbors

### 3. Stacked Generalization
- **Archivos**: `sg_breast_cancer.py`, `sg_student_performance.py`
- **Características**:
  - Arquitectura de dos niveles
  - 8 clasificadores base diversos
  - Meta-clasificador optimizado
  - Generación de meta-características
  - Análisis de correlaciones entre predictores

**Clasificadores base**:
- Random Forest
- Support Vector Machine
- Logistic Regression
- Gradient Boosting
- Naive Bayes
- K-Nearest Neighbors
- Decision Tree
- MLP Classifier

## Datasets

### 1. Breast Cancer Dataset
- **Archivo**: `dataset/breast_cancer/breast_cancer.csv`
- **Descripción**: Clasificación de tumores como malignos o benignos
- **Características**: 30 características numéricas
- **Clases**: Maligno (0), Benigno (1)
- **Muestras**: ~569 instancias

### 2. Student Performance Dataset
- **Archivo**: `dataset/student_performance/student-mat.csv`
- **Descripción**: Predicción del rendimiento académico en matemáticas
- **Características**: Variables socioeconómicas, familiares y académicas
- **Objetivo**: Aprobado (≥10) / Reprobado (<10)
- **Muestras**: ~395 estudiantes

## 📁 Estructura del Proyecto

```
classification_methods/
│
├── 📂 dataset/
│   ├── 📂 breast_cancer/
│   │   └── breast_cancer.csv
│   └── 📂 student_performance/
│       ├── student-mat.csv
│       ├── student-por.csv
│       ├── student-merge.R
│       └── student.txt
│
├── 🧠 mlp_breast_cancer.py          # MLP para breast cancer
├── 🧠 mlp_student_performance.py    # MLP para student performance
├── 🗳️ vc_breast_cancer.py           # Voting Classifier para breast cancer
├── 🗳️ vc_student_performance.py     # Voting Classifier para student performance
├── 🏗️ sg_breast_cancer.py           # Stacked Generalization para breast cancer
├── 🏗️ sg_student_performance.py     # Stacked Generalization para student performance
└── 📖 README.md
```

## Instalación

### Prerrequisitos
- Python 3.8+
- pip

### Dependencias
```bash
pip install numpy pandas scikit-learn matplotlib seaborn warnings
```

### Instalación desde GitHub
```bash
git clone https://github.com/tu-usuario/classification_methods.git
cd classification_methods
```

## Uso

### Ejecución Individual
```bash
# Perceptrón Multicapa
python mlp_breast_cancer.py
python mlp_student_performance.py

# Voting Classifier
python vc_breast_cancer.py
python vc_student_performance.py

# Stacked Generalization
python sg_breast_cancer.py
python sg_student_performance.py
```

### Ejemplo de Salida
Cada script proporciona:
- **Carga y preprocesamiento** de datos
- **Optimización de hiperparámetros**
- **Evaluación con validación cruzada**
- **Métricas en conjunto de prueba**
- **Visualizaciones comprensivas**
- **Reporte detallado de resultados**

## Resultados

### Métricas Evaluadas
- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión por clase
- **Recall**: Sensibilidad por clase
- **F1-Score**: Media armónica de precision y recall
- **ROC-AUC**: Área bajo la curva ROC (clasificación binaria)

### Visualizaciones Generadas
1. **Distribución de clases** en el dataset
2. **Matriz de confusión** con heatmap
3. **Curva ROC** con AUC
4. **Curvas de aprendizaje** para detectar overfitting
5. **Comparación entre algoritmos** (para ensembles)
6. **Análisis de meta-características** (para stacking)

### Comparación Típica de Resultados

| Dataset | Algoritmo | Accuracy | AUC-ROC | Tiempo |
|---------|-----------|----------|---------|---------|
| Breast Cancer | MLP | ~0.95 | ~0.98 | Rápido |
| Breast Cancer | Voting | ~0.96 | ~0.99 | Medio |
| Breast Cancer | Stacking | ~0.97 | ~0.99 | Lento |
| Student Perf. | MLP | ~0.85 | ~0.88 | Rápido |
| Student Perf. | Voting | ~0.87 | ~0.90 | Medio |
| Student Perf. | Stacking | ~0.88 | ~0.91 | Lento |

## Características Técnicas

### Preprocesamiento
- **Normalización**: StandardScaler para características numéricas
- **Codificación**: LabelEncoder para variables categóricas
- **División estratificada**: 80% entrenamiento / 20% prueba
- **Validación**: StratifiedKFold con 5 folds

### Optimización
- **GridSearchCV**: Búsqueda exhaustiva de hiperparámetros
- **Paralelización**: `n_jobs=-1` para máximo rendimiento
- **Early Stopping**: Prevención de overfitting en MLP
- **Validación cruzada**: Estimación robusta del rendimiento

### Manejo de Datos
- **Detección automática** de formato de archivos
- **Manejo de clases desbalanceadas**
- **Verificación de integridad** de datos
- **Conversión automática** de tipos de datos
