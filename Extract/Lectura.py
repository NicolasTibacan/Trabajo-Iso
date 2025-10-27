# ...existing code...
# ================================================================
# COMPARACIÓN ENTRE XGBOOST Y SKLEARN PARA CLASIFICACIÓN DE RIESGOS
# ================================================================

# ...existing code...
# ================================================================
# COMPARACIÓN ENTRE XGBOOST Y SKLEARN PARA CLASIFICACIÓN DE RIESGOS
# ================================================================

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle

# --- Configuración de columnas esperadas ---
required_target = "categoria_riesgo"
required_features = [
    "info_transmitida",
    "amplitud_prompt",
    "personalizacion_datos",
    "promedio",
]

# --- 1. Cargar el dataset ---
csv_path = "/workspaces/Trabajo-Iso/Extract/dataset_iso42001_operaciones.csv"
try:
    # Leer CSV con separador ';' y coma decimal
    df = pd.read_csv(csv_path, sep=";", decimal=",", encoding="utf-8")
except FileNotFoundError:
    print(f"Error: no se encontró el archivo '{csv_path}'. Coloca el CSV en el mismo directorio o ajusta la ruta.")
    sys.exit(1)
except Exception as e:
    print(f"Error leyendo CSV: {e}")
    sys.exit(1)

# Normalizar nombres de columnas (quitar espacios accidentales)
df.columns = [c.strip() for c in df.columns]

# Verificar que existan las columnas requeridas
missing = set(required_features + [required_target]) - set(df.columns)
if missing:
    print(f"Error: faltan columnas en el CSV: {sorted(list(missing))}")
    sys.exit(1)

# --- 2. Preprocesamiento ---
# Limpiar y asegurar target
df[required_target] = df[required_target].astype(str).str.strip()
# Eliminar filas con target faltante o vacías
if df[required_target].isnull().any() or (df[required_target].astype(str) == "").any():
    n_missing = int(((df[required_target].isnull()) | (df[required_target].astype(str) == "")).sum())
    print(f"Advertencia: se eliminarán {n_missing} filas con '{required_target}' vacío.")
    df = df.dropna(subset=[required_target])
    df = df[df[required_target].astype(str) != ""]

# Asegurar tipos numéricos en features (lectura ya usa decimal=',', pero coerción por si acaso)
for feat in required_features:
    # Reemplazar comas decimales dentro de strings solo si existen
    df[feat] = pd.to_numeric(df[feat].astype(str).str.replace(",", "."), errors="coerce")

# Después de conversión, comprobar si hay filas con todos los features NaN
if df[required_features].dropna(how="all").shape[0] == 0:
    print("Error: después de convertir las columnas predictoras no queda ninguna fila válida.")
    sys.exit(1)

# Codificar la variable objetivo
le = LabelEncoder()
y = le.fit_transform(df[required_target])

# Variables predictoras: tomar las columnas requeridas
X = df[required_features].copy()

# Rellenar NA simples y transformar categóricas/strings a dummies (si hubiese)
X = X.fillna(0.0)
X = pd.get_dummies(X, drop_first=True)

# Asegurar tipos numéricos (conversión segura)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Comprobar que haya features después de los dummies
if X.shape[1] == 0:
    print("Error: después de get_dummies no quedan columnas predictoras. Revisa 'required_features' y el CSV.")
    sys.exit(1)

# --- 3. División del dataset ---
# Validar que stratify sea posible (cada clase tiene al menos 2 muestras)
class_counts = pd.Series(y).value_counts()
if (class_counts < 2).any():
    print("Advertencia: hay clases con menos de 2 muestras; se omitirá 'stratify' en train_test_split.")
    stratify_arg = None
else:
    stratify_arg = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=stratify_arg
)

# --- 4. Modelo 1: RandomForest (sklearn) ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

# --- 5. Modelo 2: XGBoost ---
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss",
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)

# --- 6. Comparación de resultados ---
print("===================================================")
print("📊 RESULTADOS DE COMPARACIÓN ENTRE MODELOS")
print("===================================================")
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"XGBoost Accuracy:       {xgb_acc:.4f}")
print("===================================================")

# Mostrar informes de clasificación
labels = np.arange(len(le.classes_))
print("\n--- Reporte Random Forest ---")
print(classification_report(y_test, rf_preds, labels=labels, target_names=le.classes_))
print("\n--- Reporte XGBoost ---")
print(classification_report(y_test, xgb_preds, labels=labels, target_names=le.classes_))

# --- 7. Seleccionar el mejor modelo ---
if xgb_acc > rf_acc:
    best_model = xgb_model
    best_name = "XGBoost"
else:
    best_model = rf_model
    best_name = "RandomForest"

print(f"\n✅ El mejor modelo fue: {best_name} con accuracy = {max(xgb_acc, rf_acc):.4f}")

# --- 8. Guardar el modelo óptimo en archivo .pkl (incluye encoder y lista de features) ---
artifact = {
    "model": best_model,
    "label_encoder": le,
    "feature_columns": X_train.columns.tolist(),
}
with open("modelo_optimo_iso42001.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\n💾 Modelo y metadatos guardados exitosamente como 'modelo_optimo_iso42001.pkl'")
# ...existing code...

