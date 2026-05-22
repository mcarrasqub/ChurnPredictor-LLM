import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def main():
    print("Iniciando la generación de curvas de entrenamiento...")
    
    # Obtener el directorio raíz del proyecto (dos niveles arriba de src/evaluation)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Cargar particiones procesadas
    train_path = os.path.join(base_dir, "data", "processed", "train.csv")
    val_path = os.path.join(base_dir, "data", "processed", "val.csv")
    test_path = os.path.join(base_dir, "data", "processed", "test.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # 2. Cargar preprocesador
    preprocesador_path = os.path.join(base_dir, "models", "checkpoints", "preprocessor.joblib")
    if not os.path.exists(preprocesador_path):
        raise FileNotFoundError(f"No se encontró el preprocesador en: {preprocesador_path}")
    preproc = joblib.load(preprocesador_path)
    
    # 3. Separar X e y
    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    X_val = val_df.drop(columns=["Churn"])
    y_val = val_df["Churn"]
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]
    
    # 4. Transformar datos
    X_train_trans = preproc.transform(X_train)
    X_val_trans = preproc.transform(X_val)
    X_test_trans = preproc.transform(X_test)
    
    # 5. Entrenar el XGBoost con los mejores parámetros del GridSearch
    # Mejores parámetros: {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100, 'scale_pos_weight': 3}
    print("Ajustando modelo XGBoost para registrar la historia de aprendizaje...")
    xgb_model = XGBClassifier(
        learning_rate=0.05,
        max_depth=5,
        n_estimators=100,
        scale_pos_weight=3,
        eval_metric=["logloss", "error"],
        random_state=42
    )
    
    # Entrenar y registrar curvas
    xgb_model.fit(
        X_train_trans, y_train,
        eval_set=[(X_train_trans, y_train), (X_val_trans, y_val)],
        verbose=False
    )
    
    # Obtener resultados
    evals_result = xgb_model.evals_result()
    train_loss = evals_result['validation_0']['logloss']
    val_loss = evals_result['validation_1']['logloss']
    
    # Error de clasificación (la métrica es la proporción de clasificaciones incorrectas)
    train_error = evals_result['validation_0']['error']
    val_error = evals_result['validation_1']['error']
    
    # Transformar a exactitud (Accuracy = 1 - error)
    train_acc = [1 - err for err in train_error]
    val_acc = [1 - err for err in val_error]
    
    # 6. Graficar y guardar
    print("Graficando curvas de aprendizaje...")
    epochs = len(train_loss)
    x_axis = range(1, epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Log Loss (Pérdida)
    ax1.plot(x_axis, train_loss, label='Entrenamiento', color='#003D79', linewidth=2)
    ax1.plot(x_axis, val_loss, label='Validación', color='#FF7F0E', linewidth=2, linestyle='--')
    ax1.set_title('Función de Pérdida (Log Loss)', fontsize=12, fontweight='bold', color='#003D79')
    ax1.set_xlabel('Árboles (Iteraciones)', fontsize=10)
    ax1.set_ylabel('Log Loss', fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2: Exactitud (Accuracy)
    ax2.plot(x_axis, train_acc, label='Entrenamiento', color='#003D79', linewidth=2)
    ax2.plot(x_axis, val_acc, label='Validación', color='#FF7F0E', linewidth=2, linestyle='--')
    ax2.set_title('Exactitud (Accuracy)', fontsize=12, fontweight='bold', color='#003D79')
    ax2.set_xlabel('Árboles (Iteraciones)', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle('Curvas de Aprendizaje del Modelo XGBoost Optimizado', fontsize=14, fontweight='bold', color='#333333', y=0.98)
    plt.tight_layout()
    
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "train_val_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Curvas de aprendizaje guardadas exitosamente en: {plot_path}")
    
    # 7. Evaluar y reportar métricas del conjunto de prueba
    print("\n--- Evaluación en el Conjunto de Prueba ---")
    
    # XGBoost
    y_pred_xgb = xgb_model.predict(X_test_trans)
    y_prob_xgb = xgb_model.predict_proba(X_test_trans)[:, 1]
    
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    prec_xgb = precision_score(y_test, y_pred_xgb)
    rec_xgb = recall_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)
    
    print("\nMétricas del Modelo XGBoost Optimizado (Test Set):")
    print(f"Accuracy:  {acc_xgb:.4f}")
    print(f"Precision: {prec_xgb:.4f}")
    print(f"Recall:    {rec_xgb:.4f}")
    print(f"F1-Score:  {f1_xgb:.4f}")
    print(f"ROC-AUC:   {auc_xgb:.4f}")
    
    # Logistic Regression
    lr_path = os.path.join(base_dir, "models", "checkpoints", "lr_baseline.joblib")
    if os.path.exists(lr_path):
        lr_model = joblib.load(lr_path)
        y_pred_lr = lr_model.predict(X_test_trans)
        y_prob_lr = lr_model.predict_proba(X_test_trans)[:, 1]
        
        acc_lr = accuracy_score(y_test, y_pred_lr)
        prec_lr = precision_score(y_test, y_pred_lr)
        rec_lr = recall_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        auc_lr = roc_auc_score(y_test, y_prob_lr)
        
        print("\nMétricas de la Regresión Logística (Baseline - Test Set):")
        print(f"Accuracy:  {acc_lr:.4f}")
        print(f"Precision: {prec_lr:.4f}")
        print(f"Recall:    {rec_lr:.4f}")
        print(f"F1-Score:  {f1_lr:.4f}")
        print(f"ROC-AUC:   {auc_lr:.4f}")

if __name__ == "__main__":
    main()
