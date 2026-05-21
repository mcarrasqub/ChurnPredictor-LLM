import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Añadir el directorio raíz al path para importar src
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.agents.ollama_explain import get_shap_explanation, generate_llm_explanation

# Configuración de la página
st.set_page_config(
    page_title="ChurnPredictor-LLM | Retención de Clientes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para una estética premium (Glassmorphism & Dark Mode)
st.markdown("""
<style>
    /* Estilo general */
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Contenedores premium */
    .card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        margin-bottom: 20px;
    }
    
    /* Encabezados */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Títulos de sección */
    .section-title {
        border-left: 4px solid #00f2fe;
        padding-left: 12px;
        font-weight: 600;
        margin-bottom: 20px;
    }
    
    /* Probabilidad de fuga badge */
    .badge-high {
        background-color: #ff4b4b;
        color: white;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 14px;
    }
    .badge-low {
        background-color: #00cc96;
        color: white;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Directorios de datos y modelos
checkpoints_dir = os.path.join(base_dir, "models", "checkpoints")
test_data_path = os.path.join(base_dir, "data", "processed", "test.csv")
raw_data_path = os.path.join(base_dir, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

@st.cache_resource
def load_models():
    """Carga el modelo y preprocesador entrenados."""
    try:
        preprocesador = joblib.load(os.path.join(checkpoints_dir, "preprocessor.joblib"))
        xgb_model = joblib.load(os.path.join(checkpoints_dir, "xgb_model.joblib"))
        return preprocesador, xgb_model
    except Exception as e:
        return None, None

@st.cache_data
def load_test_data():
    """Carga los datos de prueba."""
    if os.path.exists(test_data_path):
        return pd.read_csv(test_data_path)
    elif os.path.exists(raw_data_path):
        # Carga fallback si no están procesados
        from src.data.data_loader import load_data
        df = load_data(raw_data_path)
        return df
    return None

# Título y subtítulo principal
st.markdown("""
<div style='text-align: center; padding: 10px 0px 30px 0px;'>
    <h1 style='font-size: 2.8rem; background: linear-gradient(45deg, #00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        📊 ChurnPredictor-LLM
    </h1>
    <p style='font-size: 1.1rem; color: #a0a0a0;'>
        Detección inteligente de riesgo de fuga de clientes con explicabilidad SHAP y recomendaciones LLM locales (Ollama)
    </p>
</div>
""", unsafe_allow_html=True)

# Cargar recursos
preprocesador, model = load_models()
df_test = load_test_data()

if preprocesador is None or model is None:
    st.error("⚠️ No se encontraron los modelos pre-entrenados. Por favor, asegúrate de correr primero el notebook `03_modeling.ipynb` o entrenar los modelos para generar los checkpoints.")
    st.info("Para entrenar el modelo de inmediato, puedes ejecutar en la terminal: `python -c \"from src.data.data_loader import load_data; from src.data.preprocessing import split_and_preprocess; from src.models.model_pipeline import train_models; df = load_data(); X_tr, X_va, X_te, y_tr, y_va, y_te, prep = split_and_preprocess(df); train_models(X_tr, y_tr, X_va, y_va, prep)\"`")
    st.stop()

# Configuración Sidebar
st.sidebar.markdown("### ⚙️ Configuración")
model_option = st.sidebar.selectbox(
    "Modelo LLM (Ollama)",
    ["mistral", "phi3", "llama3"],
    index=0,
    help="Modelo local previamente descargado en Ollama."
)

input_mode = st.sidebar.radio(
    "Modo de Entrada",
    ["Seleccionar Cliente Existente", "Simular Nuevo Cliente"]
)

# Cargar cliente según modo
client_data = None
if input_mode == "Seleccionar Cliente Existente":
    if df_test is not None:
        st.sidebar.markdown("### 👤 Seleccionar Cliente")
        client_idx = st.sidebar.number_input(
            f"Índice de cliente de prueba (0 - {len(df_test)-1})",
            min_value=0,
            max_value=len(df_test)-1,
            value=0
        )
        # Extraer fila de cliente y separar la columna objetivo
        client_row = df_test.iloc[[client_idx]].copy()
        real_churn = client_row["Churn"].values[0] if "Churn" in client_row.columns else None
        client_data = client_row.drop(columns=["Churn"], errors="ignore")
    else:
        st.sidebar.warning("No se encontró el conjunto de prueba. Simula un nuevo cliente.")
        input_mode = "Simular Nuevo Cliente"

if input_mode == "Simular Nuevo Cliente":
    st.sidebar.markdown("### ✏️ Características del Cliente")
    
    # Inputs para simulación
    tenure = st.sidebar.slider("Antigüedad (meses)", 1, 72, 12)
    monthly_charges = st.sidebar.slider("Cargos Mensuales ($)", 18.0, 120.0, 70.0)
    total_charges = st.sidebar.number_input("Cargos Totales ($)", min_value=18.0, max_value=8600.0, value=tenure * monthly_charges)
    
    contract = st.sidebar.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
    internet_service = st.sidebar.selectbox("Servicio de Internet", ["Fiber optic", "DSL", "No"])
    tech_support = st.sidebar.selectbox("Soporte Técnico", ["No", "Yes", "No internet service"])
    online_security = st.sidebar.selectbox("Seguridad Online", ["No", "Yes", "No internet service"])
    payment_method = st.sidebar.selectbox("Método de Pago", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
    gender = st.sidebar.selectbox("Género", ["Female", "Male"])
    senior_citizen = st.sidebar.selectbox("Adulto Mayor (Senior)", [0, 1])
    partner = st.sidebar.selectbox("Tiene Pareja", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Tiene Dependientes", ["No", "Yes"])
    phone_service = st.sidebar.selectbox("Servicio Telefónico", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Múltiples Líneas", ["No", "Yes", "No phone service"])
    online_backup = st.sidebar.selectbox("Copia Seguridad Online", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Protección de Dispositivo", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Películas", ["No", "Yes", "No internet service"])
    paperless_billing = st.sidebar.selectbox("Facturación Electrónica", ["Yes", "No"])

    # Crear dataframe de cliente simulado con estructura idéntica a X
    client_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])
    real_churn = None

# ---- PROCESAMIENTO Y PREDICCIÓN ----
if client_data is not None:
    # 1. Transformar y Predecir probabilidad
    client_trans = preprocesador.transform(client_data)
    prob_churn = model.predict_proba(client_trans)[0, 1]
    pred_churn = model.predict(client_trans)[0]

    # Distribución en columnas principales
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<h3 class='section-title'>👤 Perfil del Cliente</h3>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        # Mostrar características principales de forma estructurada
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"**Antigüedad:** {client_data['tenure'].values[0]} meses")
            st.markdown(f"**Contrato:** {client_data['Contract'].values[0]}")
            st.markdown(f"**Internet:** {client_data['InternetService'].values[0]}")
            st.markdown(f"**Soporte Técnico:** {client_data['TechSupport'].values[0]}")
        with col_c2:
            st.markdown(f"**Cargos Mensuales:** ${client_data['MonthlyCharges'].values[0]:.2f}")
            st.markdown(f"**Cargos Totales:** ${client_data['TotalCharges'].values[0]:.2f}")
            st.markdown(f"**Método de Pago:** {client_data['PaymentMethod'].values[0]}")
            st.markdown(f"**Adulto Mayor:** {'Sí' if client_data['SeniorCitizen'].values[0] == 1 else 'No'}")
        
        if real_churn is not None:
            estado_real = "Abandonó (Churn = Yes)" if real_churn == 1 else "Permaneció (Churn = No)"
            st.markdown(f"**Estado Real en Dataset:** `{estado_real}`")
            
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<h3 class='section-title'>🔮 Predicción del Modelo</h3>", unsafe_allow_html=True)
        st.markdown("<div class='card' style='text-align: center;'>", unsafe_allow_html=True)
        
        # Visualización de la Probabilidad
        st.markdown(f"#### Probabilidad de Churn (Fuga)")
        
        # Color dinámico de la predicción
        color_gauge = "#ff4b4b" if prob_churn >= 0.5 else "#00cc96"
        st.markdown(f"<h1 style='color: {color_gauge}; font-size: 3.5rem; margin: 10px 0;'>{prob_churn*100:.1f}%</h1>", unsafe_allow_html=True)
        
        # Barra de progreso
        st.progress(float(prob_churn))
        
        if prob_churn >= 0.5:
            st.markdown("<span class='badge-high'>ALTO RIESGO DE FUGA</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-low'>CLIENTE FIEL / RIESGO BAJO</span>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 class='section-title'>🎯 Explicabilidad Local (SHAP)</h3>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        # Calcular SHAP
        top_factors = get_shap_explanation(model, preprocesador, client_data)
        
        # Pintar gráfico de barras horizontal de SHAP
        fig, ax = plt.subplots(figsize=(6, 4))
        # Ordenar factores para pintar de mayor a menor importancia
        sorted_factors = sorted(top_factors.items(), key=lambda x: x[1], reverse=False)
        names = [x[0] for x in sorted_factors]
        values = [x[1] for x in sorted_factors]
        
        # Asignar color según el impacto de Churn (positivo es riesgo, negativo es retención)
        colors = ['#ff4b4b' if v > 0 else '#00cc96' for v in values]
        
        bars = ax.barh(names, values, color=colors)
        
        # Estilos premium del gráfico
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#888888')
        ax.spines['bottom'].set_color('#888888')
        ax.tick_params(colors='#cccccc', labelsize=10)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.title("Factores que influyen en la predicción (SHAP)", color='white', fontsize=12, pad=15)
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1f242d')
        plt.tight_layout()
        
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # Fila inferior: Generación de lenguaje natural con LLM (Ollama)
    st.markdown("<h3 class='section-title'>🤖 Asistente de Retención (LLM Ollama)</h3>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Botón para activar el LLM
    if st.button("Generar Resumen de Retención Personalizado (Ollama)"):
        with st.spinner(f"Consultando a {model_option} localmente..."):
            llm_text = generate_llm_explanation(top_factors, prob_churn, model_name=model_option)
            st.markdown("#### 📝 Recomendaciones de Retención:")
            st.markdown(llm_text)
    else:
        st.info("Presiona el botón superior para enviar el análisis explicativo SHAP a Ollama y obtener un reporte natural en español.")
        
    st.markdown("</div>", unsafe_allow_html=True)
