# Guía de Usuario: ChurnPredictor-LLM

Esta guía explica paso a paso cómo interactuar con el Dashboard de Predicción y Retención de Clientes.

---

## 🛠️ Requisitos Previos

Antes de ejecutar la aplicación, asegúrate de cumplir con lo siguiente:

1. **Instalar Dependencias:**
   Ejecuta en tu consola:
   ```bash
   pip install -r requirements.txt
   ```
2. **Instalar Ollama:**
   Descarga e instala Ollama desde su sitio web oficial: [https://ollama.com](https://ollama.com).
3. **Descargar el modelo Mistral:**
   Abre una terminal y ejecuta el siguiente comando para descargar el LLM en local:
   ```bash
   ollama run mistral
   ```
   *(Una vez descargado e inicializado, puedes cerrar el chat interactivo en la terminal usando `/exit`).*
4. **Tener los Modelos Entrenados:**
   Asegúrate de que los archivos `preprocessor.joblib` y `xgb_model.joblib` existan bajo `models/checkpoints/`. Si no existen, ejecuta el notebook de entrenamiento o corre en consola:
   ```bash
   python -c "from src.data.data_loader import load_data; from src.data.preprocessing import split_and_preprocess; from src.models.model_pipeline import train_models; df = load_data(); X_tr, X_va, X_te, y_tr, y_va, y_te, prep = split_and_preprocess(df); train_models(X_tr, y_tr, X_va, y_va, prep)"
   ```

---

## 🚀 Cómo Iniciar el Dashboard

Para iniciar el servidor de Streamlit y abrir la aplicación web, ejecuta el siguiente comando en la raíz del proyecto:

```bash
streamlit run app/main.py
```

La aplicación se abrirá automáticamente en tu navegador web en la dirección: `http://localhost:8501`.

---

## 🖥️ Estructura del Dashboard y Uso

### 1. Panel de Configuración Lateral (Sidebar)
*   **Modelo LLM (Ollama):** Permite elegir qué modelo local de Ollama se utilizará para el análisis y las recomendaciones (`mistral`, `phi3`, `llama3`).
*   **Modo de Entrada:**
    *   **Seleccionar Cliente Existente:** Permite ingresar el índice numérico de un cliente del conjunto de datos de prueba (`test.csv`) para predecir y explicar su comportamiento histórico.
    *   **Simular Nuevo Cliente:** Habilita controles interactivos (deslizadores y selectores) para simular en tiempo real las características de un cliente hipotético y evaluar su riesgo de fuga de inmediato.

### 2. Panel Principal

*   **Perfil del Cliente:** Muestra las variables demográficas y de servicio más relevantes del cliente cargado o simulado (como su antigüedad en meses, cargos mensuales, tipo de contrato y servicios contratados).
*   **Predicción del Modelo:** Calcula el porcentaje de probabilidad de fuga (Churn) que asigna el modelo XGBoost entrenado. Si la probabilidad es mayor o igual a 50%, se mostrará una alerta roja de **ALTO RIESGO DE FUGA**; de lo contrario, aparecerá un indicador verde de **CLIENTE FIEL**.
*   **Explicabilidad Local (SHAP):** Renderiza un gráfico de barras horizontal en tiempo real que ilustra las 5 variables de mayor influencia en la predicción.
    *   Las barras **rojas** hacia la derecha indican características que aumentan la probabilidad de fuga (factores de riesgo).
    *   Las barras **verdes** hacia la izquierda representan características que aumentan la lealtad y disminuyen la probabilidad de fuga.

### 3. Asistente de Retención (LLM)
*   Al presionar el botón **"Generar Resumen de Retención Personalizado"**, el sistema compila los datos del perfil, la probabilidad del modelo y los top 5 factores SHAP traducidos al español.
*   Envía la información localmente a la API de Ollama y muestra en pantalla un resumen ejecutivo conciso del comportamiento del cliente junto a **dos propuestas de retención específicas** recomendadas por el asistente de IA.
