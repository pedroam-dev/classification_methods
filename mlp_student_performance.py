import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configurar semilla para reproducibilidad
np.random.seed(42)

print("="*60)
print("CLASIFICACIÓN STUDENT PERFORMANCE CON PERCEPTRÓN MULTICAPA (MLP)")
print("="*60)

# 1. CARGAR Y EXPLORAR LOS DATOS
print("\n1. CARGANDO DATASET DESDE ARCHIVO LOCAL...")

# Cargar dataset desde archivo local
try:
    # Intenta diferentes formatos comunes
    dataset_path = "dataset/student_performance/student-mat.csv"  # Ajusta la ruta según tu archivo
    
    # Si es CSV
    if dataset_path.endswith('.csv'):
        data_df = pd.read_csv(dataset_path, sep=';')  # CSV con separador ';'
    # Si es Excel
    elif dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
        data_df = pd.read_excel(dataset_path)
    # Si es otro formato
    else:
        data_df = pd.read_csv(dataset_path, sep=';')
    
    print(f"Dataset cargado exitosamente desde: {dataset_path}")
    print(f"Forma del dataset: {data_df.shape}")
    
    # Mostrar las primeras filas para verificar la estructura
    print("\nPrimeras 5 filas del dataset:")
    print(data_df.head())
    
    print("\nInformación del dataset:")
    print(data_df.info())
    
    # Para student performance, usaremos G3 (nota final) como variable objetivo
    # Convertiremos en clasificación binaria: Aprobado (>=10) / Reprobado (<10)
    target_col = 'G3'
    
    if target_col not in data_df.columns:
        print(f"Error: No se encontró la columna '{target_col}'")
        print(f"Columnas disponibles: {list(data_df.columns)}")
        # Usar la última columna como respaldo
        target_col = data_df.columns[-1]
        print(f"Usando la última columna como target: '{target_col}'")
    
    print(f"\nColumna objetivo identificada: '{target_col}'")
    
    # Preprocesamiento específico para student performance
    # 1. Crear variable objetivo binaria basada en G3
    threshold = 10  # Umbral para aprobar
    data_df['pass'] = (data_df[target_col] >= threshold).astype(int)
    target_col = 'pass'
    
    print(f"Variable objetivo creada: 'pass' (1=Aprobado >=10, 0=Reprobado <10)")
    print(f"Distribución de la variable objetivo:")
    print(data_df[target_col].value_counts())
    
    # 2. Eliminar columnas que no son útiles para la predicción
    columns_to_drop = ['G1', 'G2', 'G3']  # Eliminar otras notas para evitar data leakage
    
    # 3. Codificar variables categóricas
    categorical_columns = data_df.select_dtypes(include=['object']).columns
    
    # Crear copia para procesamiento
    data_processed = data_df.copy()
    
    # Aplicar Label Encoding a variables categóricas
    label_encoders = {}
    for col in categorical_columns:
        if col not in columns_to_drop and col != target_col:
            le = LabelEncoder()
            data_processed[col] = le.fit_transform(data_processed[col])
            label_encoders[col] = le
            print(f"Codificada variable categórica: {col}")
    
    # Eliminar columnas no necesarias
    features_df = data_processed.drop(columns=columns_to_drop + [target_col])
    
    # Separar características y variable objetivo
    X = features_df.values
    y = data_processed[target_col].values
    
    print(f"\nCaracterísticas utilizadas: {list(features_df.columns)}")
    print(f"Número de características después del preprocesamiento: {X.shape[1]}")
    
    # Crear nombres de clases
    target_names = ['Reprobado', 'Aprobado']
    
    # Crear objeto similar al de sklearn para compatibilidad
    class DataContainer:
        def __init__(self, data, target, target_names, feature_names):
            self.data = data
            self.target = target
            self.target_names = target_names
            self.feature_names = feature_names
    
    feature_names = list(features_df.columns)
    data = DataContainer(X, y, target_names, feature_names)
    
except FileNotFoundError:
    print(f"Error: No se pudo encontrar el archivo '{dataset_path}'")
    print("Por favor, verifica la ruta del archivo.")
    exit()
except Exception as e:
    print(f"Error al cargar el dataset: {e}")
    exit()

print(f"Forma del dataset: {X.shape}")
print(f"Número de características: {X.shape[1]}")
print(f"Número de muestras: {X.shape[0]}")
print(f"Clases: {data.target_names}")
print(f"Distribución de clases: {np.bincount(y)}")

# 2. PREPROCESAMIENTO DE DATOS
print("\n2. PREPROCESANDO DATOS...")

# Verificar si hay suficientes muestras de cada clase para estratificación
class_counts = np.bincount(y)
min_class_size = min(class_counts)
print(f"Tamaño mínimo de clase: {min_class_size}")

if min_class_size < 2:
    print("Advertencia: Una clase tiene muy pocas muestras. Usando división aleatoria.")
    stratify_param = None
else:
    stratify_param = y

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_param
)

print(f"Tamaño conjunto entrenamiento: {X_train.shape[0]}")
print(f"Tamaño conjunto prueba: {X_test.shape[0]}")
print(f"Distribución en entrenamiento: {np.bincount(y_train)}")
print(f"Distribución en prueba: {np.bincount(y_test)}")

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
if min_class_size >= 3:  # Necesitamos al menos 3 muestras por clase para CV con 3 folds
    cv_folds = min(5, min_class_size)
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    print(f"Usando validación cruzada estratificada con {cv_folds} folds")
else:
    cv_strategy = 3  # Usar CV simple
    print("Usando validación cruzada simple debido a clases desbalanceadas")

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
print("Iniciando búsqueda de hiperparámetros...")
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

scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
if len(np.unique(y)) == 2:  # Solo para clasificación binaria
    scoring_metrics.append('roc_auc')

cv_results = {}

for metric in scoring_metrics:
    try:
        scores = cross_val_score(best_mlp, X_train_scaled, y_train, cv=cv_strategy, scoring=metric)
        cv_results[metric] = scores
        print(f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    except Exception as e:
        print(f"Error calculando {metric}: {e}")

# 7. EVALUACIÓN EN CONJUNTO DE PRUEBA
print("\n7. EVALUACIÓN EN CONJUNTO DE PRUEBA...")

# Predicciones
y_pred = best_mlp.predict(X_test_scaled)

# Verificar si el modelo puede predecir probabilidades
try:
    y_pred_proba = best_mlp.predict_proba(X_test_scaled)
    if y_pred_proba.shape[1] > 1:
        y_pred_proba = y_pred_proba[:, 1]
    else:
        y_pred_proba = y_pred_proba[:, 0]
except:
    y_pred_proba = None
    print("Nota: No se pueden calcular probabilidades para este modelo")

# Métricas de rendimiento
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy en conjunto de prueba: {test_accuracy:.4f}")

print("\n" + "="*50)
print("REPORTE DE CLASIFICACIÓN:")
print("="*50)
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 8. VISUALIZACIONES
print("\n8. GENERANDO VISUALIZACIONES...")

# Configurar estilo de gráficos más compacto
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Evaluación del Perceptrón Multicapa (MLP) - Student Performance', fontsize=14, fontweight='bold')

# Subplot 1: Distribución de clases
axes[0, 0].bar(data.target_names, np.bincount(y), color=['lightcoral', 'lightblue'])
axes[0, 0].set_title('Distribución de clases', fontsize=12)
axes[0, 0].set_ylabel('Número de muestras', fontsize=10)
axes[0, 0].tick_params(axis='both', which='major', labelsize=9)
for i, v in enumerate(np.bincount(y)):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=9)

# Subplot 2: Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=data.target_names, yticklabels=data.target_names, ax=axes[0, 1],
            cbar_kws={'shrink': 0.7}, annot_kws={'size': 10})
axes[0, 1].set_title('Matriz de confusión', fontsize=12)
axes[0, 1].set_ylabel('Valores reales', fontsize=10)
axes[0, 1].set_xlabel('Predicciones', fontsize=10)
axes[0, 1].tick_params(axis='both', which='major', labelsize=9)

# Subplot 3: Curva ROC (solo si tenemos probabilidades y es clasificación binaria)
if y_pred_proba is not None and len(np.unique(y)) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Línea base')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel('Tasa de Falsos Positivos', fontsize=10)
    axes[1, 0].set_ylabel('Tasa de Verdaderos Positivos', fontsize=10)
    axes[1, 0].set_title('Curva ROC', fontsize=12)
    axes[1, 0].legend(loc="lower right", fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=9)
else:
    # Mostrar distribución de notas originales
    axes[1, 0].hist(data_df['G3'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Umbral = {threshold}')
    axes[1, 0].set_xlabel('Nota Final (G3)', fontsize=10)
    axes[1, 0].set_ylabel('Frecuencia', fontsize=10)
    axes[1, 0].set_title('Distribución de Notas Finales', fontsize=12)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Curvas de aprendizaje
try:
    train_sizes, train_scores, val_scores = learning_curve(
        best_mlp, X_train_scaled, y_train, cv=cv_strategy, 
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    axes[1, 1].plot(train_sizes, train_mean, 'o-', color='blue', label='Entrenamiento', markersize=4)
    axes[1, 1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    axes[1, 1].plot(train_sizes, val_mean, 'o-', color='red', label='Validación', markersize=4)
    axes[1, 1].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    axes[1, 1].set_xlabel('Tamaño del conjunto de entrenamiento', fontsize=10)
    axes[1, 1].set_ylabel('Accuracy', fontsize=10)
    axes[1, 1].set_title('Curvas de Aprendizaje', fontsize=12)
    axes[1, 1].legend(loc='lower right', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=9)
except Exception as e:
    print(f"Error generando curvas de aprendizaje: {e}")
    axes[1, 1].text(0.5, 0.5, 'Error generando\ncurvas de aprendizaje', 
                   ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)

# Ajustar espaciado para gráfica más compacta
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Análisis de características más importantes (si es posible)
if hasattr(best_mlp, 'coefs_'):
    print("\n" + "="*50)
    print("ANÁLISIS DE CARACTERÍSTICAS:")
    print("="*50)
    
    # Calcular importancia aproximada basada en los pesos de la primera capa
    feature_importance = np.abs(best_mlp.coefs_[0]).mean(axis=1)
    
    # Crear DataFrame con importancias
    importance_df = pd.DataFrame({
        'feature': data.feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("Top 10 características más importantes:")
    print(importance_df.head(10))
    
    # Visualizar importancia de características
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importancia (peso promedio)')
    plt.title('Top 15 Características Más Importantes')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# 9. RESUMEN FINAL
print("\n" + "="*60)
print("RESUMEN DE RESULTADOS:")
print("="*60)

print(f"Algoritmo: Perceptrón Multicapa (MLP)")
print(f"Dataset: Student Performance (Matemáticas)")
print(f"Tarea: Clasificación binaria (Aprobado/Reprobado)")
print(f"Umbral de aprobación: {threshold}")
print(f"Arquitectura: {best_mlp.hidden_layer_sizes}")
print(f"Función de activación: {best_mlp.activation}")
print(f"Solver: {best_mlp.solver}")
print(f"Regularización (alpha): {best_mlp.alpha}")
print(f"Accuracy en validación cruzada: {grid_search.best_score_:.4f}")
print(f"Accuracy en conjunto de prueba: {test_accuracy:.4f}")

if y_pred_proba is not None and len(np.unique(y)) == 2:
    print(f"AUC-ROC: {roc_auc:.4f}")

# Información adicional del modelo
print(f"\nInformación del modelo entrenado:")
print(f"Número de iteraciones: {best_mlp.n_iter_}")
print(f"Número de capas: {best_mlp.n_layers_}")
print(f"Número de salidas: {best_mlp.n_outputs_}")

# Estadísticas del dataset
print(f"\nEstadísticas del dataset:")
print(f"Total de estudiantes: {len(y)}")
print(f"Estudiantes aprobados: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
print(f"Estudiantes reprobados: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
print(f"Nota promedio: {data_df['G3'].mean():.2f}")
print(f"Nota mediana: {data_df['G3'].median():.2f}")

print("\n" + "="*60)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("="*60)