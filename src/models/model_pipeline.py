import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

def train_models(X_train, y_train, X_val, y_val, preprocesador, checkpoints_dir=None):
    """
    Entrena el baseline de Regresión Logística y optimiza XGBoost mediante GridSearchCV.
    - Guarda el clasificador XGBoost entrenado en models/checkpoints/.
    - Retorna el modelo XGBoost y el baseline entrenados.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(base_dir, "models", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 1. Transformar datos para entrenamiento directo en clasificadores
    X_train_trans = preprocesador.transform(X_train)
    X_val_trans = preprocesador.transform(X_val)
    
    # --- Regresión Logística (Baseline) ---
    print("\n--- Entrenando Baseline: Regresión Logística ---")
    baseline_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    baseline_lr.fit(X_train_trans, y_train)
    
    # Evaluar baseline en validación
    preds_lr = baseline_lr.predict(X_val_trans)
    probs_lr = baseline_lr.predict_proba(X_val_trans)[:, 1]
    print(classification_report(y_val, preds_lr))
    print(f"Baseline ROC-AUC: {roc_auc_score(y_val, probs_lr):.4f}")
    
    # --- XGBoost con Tuning Básico (GridSearchCV) ---
    print("\n--- Optimizando Modelo Principal: XGBoost (GridSearchCV) ---")
    xgb_base = XGBClassifier(eval_metric="logloss", random_state=42)
    
    # Grilla de parámetros reducida para velocidad y efectividad
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'scale_pos_weight': [3]  # Para manejar el desbalance de clases (relación 3:1)
    }
    
    # GridSearchCV optimizando F1-Score (prioritario para el negocio)
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_trans, y_train)
    best_xgb = grid_search.best_estimator_
    
    print(f"\nMejores parámetros encontrados: {grid_search.best_params_}")
    
    # Evaluar XGBoost optimizado en validación
    preds_xgb = best_xgb.predict(X_val_trans)
    probs_xgb = best_xgb.predict_proba(X_val_trans)[:, 1]
    print("\nReporte de Clasificación (XGBoost Optimizado en Validación):")
    print(classification_report(y_val, preds_xgb))
    print(f"XGBoost ROC-AUC: {roc_auc_score(y_val, probs_xgb):.4f}")
    
    # 3. Guardar el clasificador XGBoost y la Regresión Logística
    joblib.dump(best_xgb, os.path.join(checkpoints_dir, "xgb_model.joblib"))
    joblib.dump(baseline_lr, os.path.join(checkpoints_dir, "lr_baseline.joblib"))
    print(f"Modelos guardados exitosamente en: {checkpoints_dir}")
    
    return best_xgb, baseline_lr
