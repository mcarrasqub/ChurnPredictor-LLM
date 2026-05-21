import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def split_and_preprocess(df, output_dir=None):
    """
    Divide los datos de forma estratificada en Train/Val/Test y define el preprocesador.
    - Guarda train.csv, val.csv y test.csv en data/processed/.
    - Ajusta el preprocesador en Train y lo guarda en models/checkpoints/.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if output_dir is None:
        output_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoints_dir = os.path.join(base_dir, "models", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 1. Separación de X e y
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # 2. Partición estratificada (64% Train, 16% Val, 20% Test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    # Guardar conjuntos divididos en data/processed/ (con variable objetivo incluida)
    train_df = X_train.copy()
    train_df["Churn"] = y_train
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    
    val_df = X_val.copy()
    val_df["Churn"] = y_val
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    
    test_df = X_test.copy()
    test_df["Churn"] = y_test
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Particiones guardadas en: {output_dir}")
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    
    # 3. Detección de columnas
    columnas_num = X.select_dtypes(include="number").columns.tolist()
    columnas_cat = X.select_dtypes(include=["object", "str"]).columns.tolist()
    
    # 4. Definición del Preprocesador
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
    
    # Ajustar preprocesador en entrenamiento
    preprocesador.fit(X_train)
    
    # Guardar el preprocesador ajustado
    preprocessor_path = os.path.join(checkpoints_dir, "preprocessor.joblib")
    joblib.dump(preprocesador, preprocessor_path)
    print(f"Preprocesador guardado en: {preprocessor_path}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocesador
