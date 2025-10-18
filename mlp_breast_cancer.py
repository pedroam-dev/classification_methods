import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configurar semilla para reproducibilidad
np.random.seed(42)

print("="*60)
print("CLASIFICACIÓN BREAST CANCER CON PERCEPTRÓN MULTICAPA (MLP)")
print("="*60)

# 1. CARGAR Y EXPLORAR LOS DATOS
print("\n1. CARGANDO DATASET DESDE ARCHIVO LOCAL...")

# Cargar dataset desde archivo local
# Ajusta la ruta según donde tengas tu archivo
try:
    # Intenta diferentes formatos comunes
    dataset_path = "dataset/breast_cancer/breast_cancer.csv"  # Ajusta la ruta según tu archivo
    
    # Si es CSV
    if dataset_path.endswith('.csv'):
        data_df = pd.read_csv(dataset_path)
    # Si es Excel
    elif dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
        data_df = pd.read_excel(dataset_path)
    # Si es otro formato
    else:
        data_df = pd.read_csv(dataset_path)
    
    print(f"Dataset cargado exitosamente desde: {dataset_path}")
    print(f"Forma del dataset: {data_df.shape}")
    
    # Mostrar las primeras filas para verificar la estructura
    print("\nPrimeras 5 filas del dataset:")
    print(data_df.head())
    
    print("\nInformación del dataset:")
    print(data_df.info())
    
    # Identificar columna objetivo (ajusta según tu dataset)
    # Asume que la última columna es el target o busca columnas comunes
    target_columns = ['target', 'class', 'diagnosis', 'label', 'y']
    target_col = None
    
    for col in target_columns:
        if col in data_df.columns:
            target_col = col
            break
    
    if target_col is None:
        # Si no encuentra columna objetivo automáticamente, usa la última columna
        target_col = data_df.columns[-1]
        print(f"\nUsando la última columna como target: '{target_col}'")
    else:
        print(f"\nColumna objetivo identificada: '{target_col}'")
    
    # Separar características y variable objetivo
    X = data_df.drop(columns=[target_col]).values
    y = data_df[target_col].values
    
    # Crear nombres de clases si no están disponibles
    unique_classes = np.unique(y)
    if len(unique_classes) == 2:
        if set(unique_classes) == {0, 1}:
            target_names = ['Maligno', 'Benigno']
        elif set(unique_classes) == {'M', 'B'}:
            # Convertir M/B a 0/1
            y = np.where(y == 'M', 0, 1)
            target_names = ['Maligno', 'Benigno']
        else:
            target_names = [f'Clase_{cls}' for cls in unique_classes]
    else:
        target_names = [f'Clase_{cls}' for cls in unique_classes]
    
    # Crear objeto similar al de sklearn para compatibilidad
    class DataContainer:
        def __init__(self, data, target, target_names, feature_names):
            self.data = data
            self.target = target
            self.target_names = target_names
            self.feature_names = feature_names
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    if 'feature_names' in data_df.columns or len(data_df.columns) > 2:
        feature_names = [col for col in data_df.columns if col != target_col]
    
    data = DataContainer(X, y, target_names, feature_names)
    
except FileNotFoundError:
    print(f"Error: No se pudo encontrar el archivo '{dataset_path}'")
    print("Usando dataset de sklearn como respaldo...")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
except Exception as e:
    print(f"Error al cargar el dataset: {e}")
    print("Usando dataset de sklearn como respaldo...")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target

print(f"Forma del dataset: {X.shape}")
print(f"Número de características: {X.shape[1]}")
print(f"Número de muestras: {X.shape[0]}")
print(f"Clases: {data.target_names}")
print(f"Distribución de clases: {np.bincount(y)}")

# 2. PREPROCESAMIENTO DE DATOS
print("\n2. PREPROCESANDO DATOS...")

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Tamaño conjunto entrenamiento: {X_train.shape[0]}")
print(f"Tamaño conjunto prueba: {X_test.shape[0]}")

# Normalizar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Datos normalizados correctamente.")

# 3. DEFINIR PARÁMETROS PARA OPTIMIZACIÓN
print("\n3. CONFIGURANDO OPTIMIZACIÓN DE HIPERPARÁMETROS...")

# Definir grid de parámetros para búsqueda
param_grid = {
    'hidden_layer_sizes': [
        (50,), (100,), (150,),           # Una capa oculta
        (50, 25), (100, 50), (150, 75),  # Dos capas ocultas
        (100, 50, 25)                    # Tres capas ocultas
    ],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'lbfgs'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [500, 1000]
}

# 4. OPTIMIZACIÓN CON VALIDACIÓN CRUZADA
print("\n4. EJECUTANDO BÚSQUEDA DE HIPERPARÁMETROS...")

# Configurar validación cruzada estratificada
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Crear modelo base
mlp = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)

# Búsqueda en grid con validación cruzada
grid_search = GridSearchCV(
    mlp, param_grid, 
    cv=cv_strategy, 
    scoring='accuracy', 
    n_jobs=-1, 
    verbose=1
)

# Entrenar con búsqueda de hiperparámetros
grid_search.fit(X_train_scaled, y_train)

print(f"\nMejores parámetros encontrados:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Mejor score de validación cruzada: {grid_search.best_score_:.4f}")

# 5. ENTRENAR MODELO FINAL CON MEJORES PARÁMETROS
print("\n5. ENTRENANDO MODELO FINAL...")

best_mlp = grid_search.best_estimator_
best_mlp.fit(X_train_scaled, y_train)

# 6. VALIDACIÓN CRUZADA CON MÚLTIPLES MÉTRICAS
print("\n6. EVALUACIÓN CON VALIDACIÓN CRUZADA...")

scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = {}

for metric in scoring_metrics:
    scores = cross_val_score(best_mlp, X_train_scaled, y_train, cv=cv_strategy, scoring=metric)
    cv_results[metric] = scores
    print(f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# 7. EVALUACIÓN EN CONJUNTO DE PRUEBA
print("\n7. EVALUACIÓN EN CONJUNTO DE PRUEBA...")

# Predicciones
y_pred = best_mlp.predict(X_test_scaled)
y_pred_proba = best_mlp.predict_proba(X_test_scaled)[:, 1]

# Métricas de rendimiento
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy en conjunto de prueba: {test_accuracy:.4f}")

print("\n" + "="*50)
print("REPORTE DE CLASIFICACIÓN:")
print("="*50)
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 8. VISUALIZACIONES
print("\n8. GENERANDO VISUALIZACIONES...")

# Configurar estilo de gráficos
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Evaluación del Perceptrón Multicapa (MLP) - Breast Cancer', fontsize=16, fontweight='bold')

# Subplot 1: Distribución de clases
axes[0, 0].bar(data.target_names, np.bincount(y), color=['lightcoral', 'lightblue'])
axes[0, 0].set_title('Distribución de clases en el dataset')
axes[0, 0].set_ylabel('Número de muestras')
for i, v in enumerate(np.bincount(y)):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Subplot 2: Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=data.target_names, yticklabels=data.target_names, ax=axes[0, 1])
axes[0, 1].set_title('Matriz de confusión')
axes[0, 1].set_ylabel('Valores reales')
axes[0, 1].set_xlabel('Predicciones')

# Subplot 3: Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Línea base')
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel('Tasa de Falsos Positivos')
axes[1, 0].set_ylabel('Tasa de Verdaderos Positivos')
axes[1, 0].set_title('Curva ROC')
axes[1, 0].legend(loc="lower right")
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Curvas de aprendizaje
train_sizes, train_scores, val_scores = learning_curve(
    best_mlp, X_train_scaled, y_train, cv=cv_strategy, 
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

axes[1, 1].plot(train_sizes, train_mean, 'o-', color='blue', label='Entrenamiento')
axes[1, 1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
axes[1, 1].plot(train_sizes, val_mean, 'o-', color='red', label='Validación')
axes[1, 1].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
axes[1, 1].set_xlabel('Tamaño del conjunto de entrenamiento')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Curvas de Aprendizaje')
axes[1, 1].legend(loc='lower right')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 9. RESUMEN FINAL
print("\n" + "="*60)
print("RESUMEN DE RESULTADOS:")
print("="*60)

print(f"Algoritmo: Perceptrón Multicapa (MLP)")
print(f"Arquitectura: {best_mlp.hidden_layer_sizes}")
print(f"Función de activación: {best_mlp.activation}")
print(f"Solver: {best_mlp.solver}")
print(f"Regularización (alpha): {best_mlp.alpha}")
print(f"Accuracy en validación cruzada: {grid_search.best_score_:.4f}")
print(f"Accuracy en conjunto de prueba: {test_accuracy:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Información adicional del modelo
print(f"\nInformación del modelo entrenado:")
print(f"Número de iteraciones: {best_mlp.n_iter_}")
print(f"Número de capas: {best_mlp.n_layers_}")
print(f"Número de salidas: {best_mlp.n_outputs_}")

print("\n" + "="*60)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("="*60)