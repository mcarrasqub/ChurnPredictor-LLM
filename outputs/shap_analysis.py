import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = BASE_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carga y Limpieza de Datos 
df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)
df.drop(columns=["customerID"], inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print(f"Dataset cargado correctamente: {df.shape[0]} filas.")

# Preprocesamiento y Partición de Datos
X = df.drop(columns="Churn")
y = df["Churn"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

# Selección de dtypes
columnas_num = X.select_dtypes(include="number").columns.tolist()
columnas_cat = X.select_dtypes(include=["object", "str"]).columns.tolist()

preprocesador = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), columnas_num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), columnas_cat)
])

# Entrenamiento de XGBoost 
modelo_xgb = Pipeline([
    ("prep", preprocesador),
    ("clf", XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=3,
        eval_metric="logloss",
        random_state=42
    ))
])
modelo_xgb.fit(X_train, y_train)
print("Modelo entrenado con éxito.")

# Evaluación del Modelo
y_pred_val = modelo_xgb.predict(X_val)
y_prob_val = modelo_xgb.predict_proba(X_val)[:, 1]

print("\n=== Reporte de Clasificación (Validación) ===")
print(classification_report(y_val, y_pred_val))
print("ROC-AUC:", roc_auc_score(y_val, y_prob_val))

# Generar y guardar la matriz de confusión
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(modelo_xgb, X_val, y_val, cmap="Blues", ax=plt.gca())
plt.title("Matriz de Confusión - XGBoost")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()
print(f"Matriz de confusión guardada en: {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")

# Cálculo de Explicabilidad SHAP
# Extraer estimador y preprocesador
clf = modelo_xgb.named_steps["clf"]
X_val_transformado = modelo_xgb.named_steps["prep"].transform(X_val)

# Recuperar los nombres de las columnas transformadas
feature_names = (columnas_num + 
    modelo_xgb.named_steps["prep"]
    .named_transformers_["cat"]
    .named_steps["encoder"]
    .get_feature_names_out(columnas_cat).tolist()
)

# Calcular valores SHAP
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_val_transformado)

# Generar y guardar la gráfica resumen
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_val_transformado, feature_names=feature_names, show=False)
plt.title("Importancia de Variables (SHAP)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=300)
plt.show()

# Generar diccionario para la integración con Ollama
# Extraer el impacto de variables para un cliente específico
cliente_idx = 0
shap_cliente = dict(zip(
    feature_names,
    shap_values[cliente_idx]
))

# Filtrar los 5 factores con mayor impacto absoluto
top5 = dict(sorted(shap_cliente.items(),
                   key=lambda x: abs(x[1]),
                   reverse=True)[:5])

print("\n--- Salida del SHAP para Ollama (Top 5 factores) ---")
print(top5)

# Guardar en pickle
pickle_path = os.path.join(OUTPUT_DIR, "shap_output.pkl")
with open(pickle_path, "wb") as f:
    pickle.dump(top5, f)

print(f"\n¡Proceso finalizado! Archivo pickle guardado en: {pickle_path}")
