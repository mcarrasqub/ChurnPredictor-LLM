# ChurnPredictor-LLM 📊🤖

Este proyecto implementa un flujo completo de Ciencia de Datos y Machine Learning para predecir y explicar la fuga de clientes (*Churn*) en el dataset clásico **IBM Telco Customer Churn**. Está diseñado para integrarse con modelos de lenguaje locales (LLMs) mediante **Ollama**, permitiendo generar explicaciones personalizadas y comprensibles sobre los factores de riesgo de cada cliente a través de una interfaz interactiva de **Streamlit**.

---

## 📁 Estructura del Proyecto

El repositorio está organizado bajo el siguiente estándar de desarrollo y reproducibilidad:

```text
proyecto-ia-eafit/
├── README.md                    # Instrucciones de ejecución claras (este archivo)
├── requirements.txt             # Dependencias del proyecto
│
├── docs/
│   ├── informe_final.pdf        # Informe final del proyecto (compilado)
│   └── guia_usuario.md          # Guía detallada de uso con explicaciones del sistema
│
├── data/
│   ├── raw/                     # Dataset original (WA_Fn-UseC_-Telco-Customer-Churn.csv)
│   └── processed/               # Datos limpios y particionados (train.csv, val.csv, test.csv)
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploración y análisis de datos (Persona 1)
│   ├── 02_preprocessing.ipynb   # Ingeniería de variables y preprocesamiento (Persona 1)
│   ├── 03_modeling.ipynb        # Modelado, tuning y evaluación en test set (Persona 1)
│   └── 04_llm_rag_agents.ipynb  # Integración local con Ollama (Persona 2)
│
├── src/
│   ├── data/
│   │   ├── data_loader.py       # Carga y limpieza inicial del dataset
│   │   └── preprocessing.py     # Flujo del ColumnTransformer y particiones
│   ├── models/
│   │   └── model_pipeline.py    # Definición y tuning (GridSearchCV) del XGBoost
│   ├── evaluation/
│   │   └── metrics_eval.py      # Reporte de métricas de prueba y matriz de confusión
│   └── agents/
│       └── ollama_explain.py    # Explicabilidad local SHAP y prompts en Ollama
│
├── models/
│   └── checkpoints/             # Checkpoints de preprocesador y modelo entrenado (.joblib)
│
├── outputs/
│   └── confusion_matrix.png     # Matriz de confusión final en el set de prueba
│
└── app/
    └── main.py                  # Interfaz interactiva desarrollada en Streamlit
```

---

## ⚙️ Requisitos y Configuración de Entorno

### 1. Clonar e Instalar Dependencias
Instala los requerimientos en tu entorno local de Python:
```bash
pip install -r requirements.txt
```

### 2. Configurar el LLM Local (Ollama)
Para usar el asistente de IA local para retención de clientes:
1. Descarga e instala Ollama desde [https://ollama.com](https://ollama.com).
2. Descarga el modelo de lenguaje ejecutando en tu terminal:
   ```bash
   ollama run mistral
   ```

---

## 🚀 Guía de Ejecución

### Paso 1: Ejecutar el Pipeline de Machine Learning
Para realizar la limpieza, preprocesar, entrenar (con tuning de hiperparámetros) y evaluar el modelo sobre el conjunto de prueba independiente, ejecuta el siguiente comando:

```bash
python -c "from src.data.data_loader import load_data; from src.data.preprocessing import split_and_preprocess; from src.models.model_pipeline import train_models; from src.evaluation.metrics_eval import evaluate_on_test; df = load_data(); X_train, X_val, X_test, y_train, y_val, y_test, prep = split_and_preprocess(df); model, baseline = train_models(X_train, y_train, X_val, y_val, prep); evaluate_on_test(model, prep, X_test, y_test)"
```
Este comando generará:
*   Las particiones de datos en `data/processed/`.
*   Los checkpoints `preprocessor.joblib` y `xgb_model.joblib` en `models/checkpoints/`.
*   El reporte de métricas finales y la gráfica `outputs/confusion_matrix.png`.

*Nota: También puedes ejecutar secuencialmente los notebooks numerados `01_eda.ipynb`, `02_preprocessing.ipynb` y `03_modeling.ipynb` bajo la carpeta `notebooks/`.*

### Paso 2: Lanzar la Aplicación Interactiva (Streamlit)
Una vez entrenado el modelo, inicia la aplicación web interactiva con:

```bash
streamlit run app/main.py
```
Abre en tu navegador `http://localhost:8501`. Podrás seleccionar cualquier cliente del conjunto de prueba o simular uno en tiempo real, visualizar su probabilidad de fuga, el impacto de sus variables con SHAP y generar recomendaciones automatizadas a través de Ollama.

---

## 👥 Integrantes
*   **Mariana Carrasquilla Botero** (Persona 1: Data Science & ML)
*   **Sofía Gallo de la Rosa** (Persona 2: LLM, Backend & Reproducibilidad)
