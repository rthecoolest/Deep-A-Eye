import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)


def compute_specificity_from_cm(cm):
    per_class_specificity = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class_specificity.append(float(specificity))
    return per_class_specificity, float(np.mean(per_class_specificity))


def compute_metrics(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_specificity, macro_specificity = compute_specificity_from_cm(cm)

    report_dict = classification_report(
        y_true, y_pred, labels=labels, digits=4, zero_division=0, output_dict=True
    )

    per_class = {}
    for c in labels:
        key = str(c)
        if key in report_dict:
            per_class[c] = {
                "precision": float(report_dict[key]["precision"]),
                "recall_sensitivity": float(report_dict[key]["recall"]),
                "f1": float(report_dict[key]["f1-score"]),
                "support": int(report_dict[key]["support"]),
                "specificity": float(per_class_specificity[c]) if c < len(per_class_specificity) else 0.0,
            }

    return {
        "acc": float(acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_sensitivity": float(macro_recall),
        "macro_specificity": float(macro_specificity),
        "macro_f1": float(macro_f1),
        "qwk": float(qwk),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "classification_report_text_ready": classification_report(
            y_true, y_pred, labels=labels, digits=4, zero_division=0
        )
    }


def print_metrics_block(split_name, metrics):
    print(f"\n===== {split_name.upper()} METRICS =====")
    print(f"Accuracy: {metrics['acc']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro Sensitivity: {metrics['macro_sensitivity']:.4f}")
    print(f"Macro Specificity: {metrics['macro_specificity']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    print(f"QWK: {metrics['qwk']:.4f}")

    print("\nPer-class metrics:")
    for cls, vals in metrics["per_class"].items():
        print(
            f"Class {cls} | "
            f"Precision: {vals['precision']:.4f} | "
            f"Recall/Sensitivity: {vals['recall_sensitivity']:.4f} | "
            f"Specificity: {vals['specificity']:.4f} | "
            f"F1: {vals['f1']:.4f} | "
            f"Support: {vals['support']}"
        )

    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    print("\nClassification Report:")
    print(metrics["classification_report_text_ready"])