import os
import pandas as pd

def load_data(filepath=None):
    """
    Carga el dataset de IBM Telco Customer Churn y realiza la limpieza básica.
    - Convierte TotalCharges a numérico y elimina valores nulos resultantes.
    - Remueve la columna 'customerID'.
    - Codifica la columna objetivo 'Churn' en valores binarios (1/0).
    """
    if filepath is None:
        # Ruta por defecto
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filepath = os.path.join(base_dir, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo de datos en la ruta: {filepath}")
        
    df = pd.read_csv(filepath)
    
    # 1. Limpieza de TotalCharges (vacíos a NaN y removerlos)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    
    # 2. Eliminar customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        
    # 3. Codificar variable objetivo Churn (Yes: 1, No: 0)
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        
    return df
