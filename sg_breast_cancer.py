import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configurar semilla para reproducibilidad
np.random.seed(42)

print("="*60)
print("CLASIFICACIÓN BREAST CANCER CON STACKED GENERALIZATION")
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

# 3. DEFINIR CLASIFICADORES BASE Y META-CLASIFICADOR
print("\n3. CONFIGURANDO STACKED GENERALIZATION Y OPTIMIZACIÓN DE HIPERPARÁMETROS...")

# Definir clasificadores base (nivel 0)
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('lr', LogisticRegression(random_state=42, max_iter=1000)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('nb', GaussianNB()),
    ('knn', KNeighborsClassifier()),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('mlp', MLPClassifier(random_state=42, max_iter=500))
]

# Meta-clasificadores candidatos (nivel 1)
meta_classifiers = {
    'logistic': LogisticRegression(random_state=42),
    'rf_meta': RandomForestClassifier(n_estimators=50, random_state=42),
    'svm_meta': SVC(probability=True, random_state=42),
    'gb_meta': GradientBoostingClassifier(n_estimators=50, random_state=42)
}

print(f"Clasificadores base configurados: {len(base_learners)}")
print(f"Meta-clasificadores candidatos: {len(meta_classifiers)}")

# Crear el modelo base de Stacking
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(random_state=42),
    cv=5,  # Validación cruzada interna para generar meta-características
    stack_method='predict_proba',  # Usar probabilidades como meta-características
    n_jobs=-1
)

# Definir grid de parámetros para optimización
param_grid = {
    # Parámetros de clasificadores base
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [10, 20, None],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['rbf', 'linear'],
    'lr__C': [0.1, 1, 10],
    'gb__n_estimators': [50, 100],
    'gb__learning_rate': [0.05, 0.1, 0.2],
    'knn__n_neighbors': [3, 5, 7],
    'dt__max_depth': [5, 10, 15],
    'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'mlp__activation': ['relu', 'tanh'],
    
    # Parámetros del meta-clasificador
    'final_estimator__C': [0.1, 1, 10],
    
    # Parámetros del stacking
    'cv': [3, 5],
    'stack_method': ['predict_proba', 'predict']
}

# 4. OPTIMIZACIÓN CON VALIDACIÓN CRUZADA
print("\n4. EJECUTANDO BÚSQUEDA DE HIPERPARÁMETROS...")

# Configurar validación cruzada estratificada
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Versión simplificada del grid para eficiencia computacional
param_grid_simplified = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, None],
    'svm__C': [1, 10],
    'lr__C': [1, 10],
    'gb__n_estimators': [100],
    'gb__learning_rate': [0.1],
    'knn__n_neighbors': [5],
    'dt__max_depth': [10],
    'mlp__hidden_layer_sizes': [(100,)],
    'final_estimator__C': [1],
    'cv': [5]
}

# Búsqueda en grid con validación cruzada
grid_search = GridSearchCV(
    stacking_clf, param_grid_simplified, 
    cv=cv_strategy, 
    scoring='accuracy', 
    n_jobs=-1, 
    verbose=1
)

# Entrenar con búsqueda de hiperparámetros
print("Iniciando búsqueda de hiperparámetros (esto puede tomar varios minutos)...")
grid_search.fit(X_train_scaled, y_train)

print(f"\nMejores parámetros encontrados:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Mejor score de validación cruzada: {grid_search.best_score_:.4f}")

# 5. ENTRENAR MODELO FINAL CON MEJORES PARÁMETROS
print("\n5. ENTRENANDO MODELO FINAL...")

best_stacking = grid_search.best_estimator_
best_stacking.fit(X_train_scaled, y_train)

# 6. VALIDACIÓN CRUZADA CON MÚLTIPLES MÉTRICAS
print("\n6. EVALUACIÓN CON VALIDACIÓN CRUZADA...")

scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = {}

for metric in scoring_metrics:
    scores = cross_val_score(best_stacking, X_train_scaled, y_train, cv=cv_strategy, scoring=metric)
    cv_results[metric] = scores
    print(f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# 7. EVALUACIÓN EN CONJUNTO DE PRUEBA
print("\n7. EVALUACIÓN EN CONJUNTO DE PRUEBA...")

# Predicciones
y_pred = best_stacking.predict(X_test_scaled)
y_pred_proba = best_stacking.predict_proba(X_test_scaled)[:, 1]

# Métricas de rendimiento
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy en conjunto de prueba: {test_accuracy:.4f}")

print("\n" + "="*50)
print("REPORTE DE CLASIFICACIÓN:")
print("="*50)
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Evaluación individual de clasificadores base
print("\n" + "="*50)
print("RENDIMIENTO DE CLASIFICADORES BASE:")
print("="*50)

individual_scores = {}
for name, clf in best_stacking.named_estimators_.items():
    individual_pred = clf.predict(X_test_scaled)
    individual_acc = accuracy_score(y_test, individual_pred)
    individual_scores[name] = individual_acc
    print(f"{name.upper()}: {individual_acc:.4f}")

# Evaluación del meta-clasificador
meta_features = np.column_stack([
    clf.predict_proba(X_test_scaled)[:, 1] for name, clf in best_stacking.named_estimators_.items()
])
print(f"\nMETA-CLASIFICADOR ({type(best_stacking.final_estimator_).__name__}):")
print(f"Características meta generadas: {meta_features.shape[1]}")

# 8. VISUALIZACIONES
print("\n8. GENERANDO VISUALIZACIONES...")

# Configurar estilo de gráficos más compacto
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Evaluación del Stacked Generalization - Breast Cancer', fontsize=14, fontweight='bold')

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

# Subplot 3: Curva ROC
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

# Subplot 4: Comparación de clasificadores base vs stacking
classifier_names = list(individual_scores.keys())
classifier_accuracies = list(individual_scores.values())
classifier_accuracies.append(test_accuracy)  # Añadir el stacking classifier
classifier_names.append('Stacking')

colors = plt.cm.Set3(np.linspace(0, 1, len(classifier_names)))
bars = axes[1, 1].bar(range(len(classifier_names)), classifier_accuracies, color=colors, alpha=0.8)
axes[1, 1].set_title('Comparación de Accuracy', fontsize=12)
axes[1, 1].set_ylabel('Accuracy', fontsize=10)
axes[1, 1].set_xlabel('Clasificadores', fontsize=10)
axes[1, 1].set_xticks(range(len(classifier_names)))
axes[1, 1].set_xticklabels(classifier_names, rotation=45, ha='right', fontsize=8)
axes[1, 1].tick_params(axis='y', labelsize=9)
axes[1, 1].grid(axis='y', alpha=0.3)

# Añadir valores en las barras
for i, (bar, acc) in enumerate(zip(bars, classifier_accuracies)):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

# Destacar el stacking classifier
bars[-1].set_color('gold')
bars[-1].set_edgecolor('black')
bars[-1].set_linewidth(2)

# Ajustar espaciado para gráfica más compacta
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Gráfica adicional: Curvas de aprendizaje del stacking
plt.figure(figsize=(10, 6))
train_sizes, train_scores, val_scores = learning_curve(
    best_stacking, X_train_scaled, y_train, cv=cv_strategy, 
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Entrenamiento', markersize=6)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validación', markersize=6)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
plt.xlabel('Tamaño del conjunto de entrenamiento', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Curvas de Aprendizaje - Stacked Generalization', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualización de meta-características
plt.figure(figsize=(12, 5))

# Subplot 1: Correlación entre meta-características
plt.subplot(1, 2, 1)
meta_df = pd.DataFrame(meta_features, columns=[f'Meta_{name}' for name in individual_scores.keys()])
correlation_matrix = meta_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Correlación entre Meta-Características', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Subplot 2: Distribución de meta-características
plt.subplot(1, 2, 2)
meta_df.boxplot(ax=plt.gca(), rot=45)
plt.title('Distribución de Meta-Características', fontsize=12, fontweight='bold')
plt.ylabel('Probabilidad de Clase Positiva')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. RESUMEN FINAL
print("\n" + "="*60)
print("RESUMEN DE RESULTADOS:")
print("="*60)

print(f"Algoritmo: Stacked Generalization")
print(f"Número de clasificadores base: {len(best_stacking.estimators_)}")
print(f"Meta-clasificador: {type(best_stacking.final_estimator_).__name__}")
print(f"Método de stacking: {best_stacking.stack_method}")
print(f"CV interno para meta-características: {best_stacking.cv}")
print(f"Accuracy en validación cruzada: {grid_search.best_score_:.4f}")
print(f"Accuracy en conjunto de prueba: {test_accuracy:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

print(f"\nClasificadores base incluidos:")
for name, clf in best_stacking.named_estimators_.items():
    print(f"  {name}: {type(clf).__name__} - Accuracy: {individual_scores[name]:.4f}")

print(f"\nComparación con clasificadores individuales:")
best_individual = max(individual_scores, key=individual_scores.get)
best_individual_acc = max(individual_scores.values())
print(f"Mejor clasificador individual: {best_individual} ({best_individual_acc:.4f})")
print(f"Stacked Generalization: {test_accuracy:.4f}")
improvement = test_accuracy - best_individual_acc
print(f"Mejora: {improvement:.4f} ({improvement*100:.2f}%)")

print(f"\nCaracterísticas del ensemble:")
print(f"Dimensión de meta-características: {meta_features.shape[1]}")
print(f"Diversidad de clasificadores base: {len(set([type(clf).__name__ for name, clf in best_stacking.named_estimators_.items()]))}")

print("\n" + "="*60)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("="*60)