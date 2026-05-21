import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

def evaluate_on_test(model, preprocesador, X_test, y_test, output_dir=None):
    """
    Evalúa el clasificador final en el conjunto de prueba (X_test, y_test)
    y guarda la matriz de confusión.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if output_dir is None:
        output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Transformar datos de prueba
    X_test_trans = preprocesador.transform(X_test)
    
    # 2. Predecir
    y_pred = model.predict(X_test_trans)
    y_prob = model.predict_proba(X_test_trans)[:, 1]
    
    # 3. Calcular métricas específicas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print("\n==================================================")
    print("      EVALUACIÓN EN EL CONJUNTO DE PRUEBA (TEST SET)      ")
    print("==================================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nReporte de clasificación detallado:")
    print(classification_report(y_test, y_pred))
    print("==================================================")
    
    # 4. Generar y guardar la matriz de confusión
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", ax=ax)
    plt.title("Matriz de Confusión - Conjunto de Prueba")
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Matriz de confusión de prueba guardada en: {cm_path}")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc
    }
