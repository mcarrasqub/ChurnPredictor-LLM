# ChurnPredictor-LLM 📊🤖

Este proyecto implementa un flujo completo de Ciencia de Datos y Machine Learning para predecir y explicar la fuga de clientes (*Churn*) en el dataset clásico **IBM Telco Customer Churn**. Está diseñado para integrarse con modelos de lenguaje locales (LLMs) mediante **Ollama**, permitiendo generar explicaciones personalizadas y comprensibles sobre los factores de riesgo de cada cliente.

Integrantes: Mariana Carrasquilla Botero, Sofía Gallo de la Rosa.

---

## 📁 Estructura del Proyecto

```text
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset original de IBM Telco
├── notebooks/
│   ├── eda.ipynb                              # Análisis Exploratorio de Datos (EDA)
│   └── pipeline.ipynb                         # Modelado, métricas comparativas y experimentación
├── outputs/
│   ├── shap_analysis.py                       # Script principal de entrenamiento, SHAP e integración
│   ├── shap_summary.png                       # Gráfico de importancia de variables global (SHAP)
│   ├── confusion_matrix.png                   # Matriz de confusión del modelo XGBoost seleccionado
│   └── shap_output.pkl                        # Diccionario serializado con el Top 5 de factores (para Ollama)
├── requirements.txt                           # Dependencias del proyecto
└── README.md                                  # Documentación del proyecto (este archivo)
```

---

## ⚙️ Pipeline de Preprocesamiento y Modelado

El flujo de procesamiento de datos está encapsulado en un objeto `Pipeline` de Scikit-Learn que garantiza la reproducibilidad y evita la filtración de datos (*data leakage*):
1. **Limpieza de Datos:** Conversión de la columna `TotalCharges` a tipo numérico, remoción de filas con valores nulos resultantes y eliminación de la columna identificadora `customerID`.
2. **Preprocesamiento por Tipo de Variable (`ColumnTransformer`):**
   * **Variables Numéricas:** Imputación de nulos mediante la mediana (`SimpleImputer`) y estandarización a media 0 y varianza 1 (`StandardScaler`).
   * **Variables Categóricas:** Imputación por la categoría más frecuente (`SimpleImputer`) y codificación One-Hot (`OneHotEncoder`).
3. **Manejo del Desbalanceo de Clases:** Se utiliza el parámetro `scale_pos_weight=3` en el clasificador XGBoost para compensar el desbalanceo natural del conjunto de datos (aproximadamente 3:1 a favor de los clientes activos).

---

## 📈 Resultados y Comparación de Modelos

Se compararon dos algoritmos entrenados con una partición estratificada de **Entrenamiento (64%)**, **Validación (16%)** y **Prueba (20%)**:

| Métrica (Clase Churn = 1) | Baseline: Regresión Logística | Modelo Principal: XGBoost |
| :--- | :---: | :---: |
| **Accuracy** | 0.78 | 0.75 |
| **Precision** | 0.61 | 0.52 |
| **Recall (Sensibilidad)** | 0.52 | **0.72** |
| **F1-Score** | 0.56 | **0.60** |
| **ROC-AUC** | 0.837 | 0.825 |

### 🎯 Justificación de las Métricas de Evaluación

Para este problema de negocio, **el Recall y el F1-Score son las métricas prioritarias**, por encima del Accuracy o la Precision:

1. **Costo Asimétrico de los Errores (Priorización del Recall):**
   * **Falso Negativo (FN):** Clasificamos a un cliente que realmente se va a fugar como "leal". La empresa no toma ninguna acción preventiva y el cliente abandona la compañía de manera definitiva. Esto implica perder su facturación recurrente y asumir un alto Costo de Adquisición de Cliente (CAC) para reemplazarlo.
   * **Falso Positivo (FP):** Clasificamos a un cliente leal como "propenso a la fuga". Le enviaremos una promoción o descuento de retención. Aunque esto tiene un pequeño costo marginal, es sumamente preferible antes que perder al cliente por completo.
   * **Conclusión:** Minimizar los Falsos Negativos es la meta principal del negocio, lo cual se logra **optimizando para obtener un Recall alto**.
2. **Equilibrio Financiero del Negocio (Priorización del F1-Score):**
   * Si solo optimizáramos el Recall, el modelo podría predecir perezosamente que el 100% de los clientes se van a fugar (obteniendo Recall = 1.0). Sin embargo, esto causaría que la empresa gaste recursos valiosos en campañas de retención innecesarias para clientes leales.
   * El **F1-Score** es la media armónica entre Precision y Recall. Garantiza que el modelo capture la mayor proporción de abandonos reales (Recall alto) sin deteriorar excesivamente la precisión, manteniendo la viabilidad económica de las campañas.

---

## 🤖 Integración con Ollama y Explicabilidad (SHAP)

El script `outputs/shap_analysis.py` calcula los valores de explicabilidad SHAP (*Shapley Additive exPlanations*) a nivel de cliente para identificar exactamente qué variables influyeron en su predicción.

Para la integración con **Ollama**, el script extrae los **5 factores con mayor impacto absoluto** para un cliente específico y los guarda en un archivo serializado: `outputs/shap_output.pkl`.

### 💻 ¿Cómo usar los resultados en Ollama?

Se puede cargar fácilmente el archivo de explicabilidad y construir un prompt estructurado para enviárselo a un LLM en Ollama. Aquí se ofrece un ejemplo de código en Python de cómo se haría en un entorno:

```python
import pickle
import ollama  # pip install ollama

# 1. Cargar el Top 5 de factores SHAP generados por el script
with open("outputs/shap_output.pkl", "rb") as f:
    shap_data = pickle.load(f)

# 2. Diseñar el prompt para el LLM local
prompt = f"""
Eres un analista de retención de clientes en una empresa de telecomunicaciones.
Hemos detectado que un cliente tiene un alto riesgo de cancelar su servicio (Churn). 
Los valores SHAP de explicabilidad para las 5 variables con mayor impacto en su comportamiento son:
{shap_data}

Escribe un resumen ejecutivo breve y amigable (máximo 2 párrafos) explicando las razones principales de su riesgo de fuga y sugiere 2 acciones concretas que un agente de ventas podría tomar para convencerlo de quedarse.
"""

# 3. Consultar a Ollama de manera local (usando por ejemplo 'llama3' o 'mistral')
response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": prompt}]
)

print(response["message"]["content"])
```

Con esto, el LLM de Ollama leerá los factores SHAP y generará una recomendación en lenguaje natural extremadamente útil para los agentes de retención.
