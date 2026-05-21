import os
import pickle
import shap
import numpy as np
import pandas as pd
import ollama

def clean_feature_name(feature_name):
    """
    Decodifica y limpia los nombres de variables transformados del pipeline.
    Ejemplo: 'cat__InternetService_Fiber optic' -> 'Internet Service: Fiber optic'
    """
    cleaned = feature_name
    
    # Remover prefijos de ColumnTransformer
    if cleaned.startswith("num__"):
        cleaned = cleaned.replace("num__", "")
    elif cleaned.startswith("cat__"):
        cleaned = cleaned.replace("cat__", "")
        
    # Mejorar legibilidad de variables clave
    mappings = {
        "InternetService": "Servicio de Internet",
        "Contract": "Tipo de Contrato",
        "PaymentMethod": "Método de Pago",
        "tenure": "Antigüedad (meses)",
        "MonthlyCharges": "Cargos Mensuales ($)",
        "TotalCharges": "Cargos Totales ($)",
        "OnlineSecurity": "Seguridad Online",
        "TechSupport": "Soporte Técnico",
        "PaperlessBilling": "Facturación sin papel",
        "DeviceProtection": "Protección de Dispositivo",
        "OnlineBackup": "Copia de Seguridad Online",
        "MultipleLines": "Líneas Múltiples",
        "PhoneService": "Servicio Telefónico",
        "SeniorCitizen": "Adulto Mayor"
    }
    
    for key, val in mappings.items():
        if key in cleaned:
            # Reemplazar nombre base y limpiar guiones bajos
            cleaned = cleaned.replace(key, val)
            
    # Reemplazar guiones bajos restantes por espacios
    cleaned = cleaned.replace("_", ": ").replace("  ", " ").strip()
    return cleaned

def get_shap_explanation(model, preprocesador, X_client, top_n=5):
    """
    Calcula los valores SHAP para un cliente y retorna los top N factores con nombres limpios.
    X_client debe ser un DataFrame de pandas con 1 sola fila.
    """
    # 1. Transformar características
    X_client_trans = preprocesador.transform(X_client)
    
    # 2. Recuperar nombres de características transformadas
    columnas_num = preprocesador.transformers_[0][2]
    columnas_cat = preprocesador.transformers_[1][2]
    
    encoder = preprocesador.named_transformers_["cat"].named_steps["encoder"]
    cat_features = encoder.get_feature_names_out(columnas_cat).tolist()
    
    feature_names = columnas_num + cat_features
    
    # 3. Calcular valores SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_client_trans)
    
    # SHAP retorna lista para clasificación binaria [clase_0_shap, clase_1_shap] o directo array
    # Manejar ambos formatos de salida de TreeExplainer
    if isinstance(shap_values, list):
        # Tomar los shap values de la clase positiva (1 - Churn)
        shap_values_target = shap_values[1][0]
    else:
        # Si es un solo array (para TreeExplainer con salida unidimensional)
        if len(shap_values.shape) == 3: # (num_clases, num_muestras, num_features)
            shap_values_target = shap_values[1][0]
        else: # (num_muestras, num_features)
            shap_values_target = shap_values[0]
            
    # 4. Emparejar características, valores y limpiar nombres
    shap_dict = {}
    for name, val in zip(feature_names, shap_values_target):
        clean_name = clean_feature_name(name)
        shap_dict[clean_name] = float(val)
        
    # 5. Obtener Top N factores con mayor impacto absoluto
    top_factors = dict(sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n])
    
    return top_factors

def generate_llm_explanation(top_factors, probability, model_name="mistral"):
    """
    Construye el prompt estructurado con los factores SHAP y la probabilidad de churn,
    y consulta a un LLM local mediante la API de Ollama.
    """
    # Explicación de los factores para el prompt
    factors_desc = ""
    for factor, value in top_factors.items():
        direction = "aumenta" if value > 0 else "disminuye"
        factors_desc += f"- {factor}: Impacto SHAP = {value:+.4f} ({direction} el riesgo de fuga)\n"

    prompt = f"""
Eres un analista experto en retención y fidelización de clientes en una empresa de telecomunicaciones.
Hemos detectado que un cliente tiene un riesgo de fuga del {probability * 100:.1f}%.

Los valores SHAP de explicabilidad para los 5 factores de mayor peso en su comportamiento son:
{factors_desc}

Por favor, escribe un análisis estructurado dirigido al equipo de marketing/servicio al cliente:
1. Un resumen ejecutivo amigable (máximo 2 párrafos) explicando las razones principales de su riesgo de fuga basadas en los factores anteriores.
2. Dos recomendaciones de retención específicas, viables y personalizadas según estos mismos factores.

Responde únicamente en español de forma profesional y clara.
"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error al conectar con Ollama ({model_name}): {str(e)}\n\n" \
               f"Prompt enviado:\n{prompt}"
