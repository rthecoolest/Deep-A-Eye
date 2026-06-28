import os
import torch

# ================= PATH =================
ckpt_path = "checkpoints/resnet50_uwf_late_fusion_best.pt"

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

checkpoint = torch.load(ckpt_path, map_location="cpu")

print("=" * 90)
print("CHECKPOINT INSPECTION")
print("=" * 90)

# ================= BASIC INFO =================
print("\n--- Basic Info ---")
print("Model Name:", checkpoint.get("model_name", "N/A"))
print("Run Name:", checkpoint.get("run_name", "N/A"))
print("Fusion Type:", checkpoint.get("fusion_type", "N/A"))
print("Stage:", checkpoint.get("stage", "N/A"))
print("Best Epoch:", checkpoint.get("best_epoch", "N/A"))
print("Best QWK:", checkpoint.get("best_qwk", "N/A"))

# ================= TRAIN LOSS =================
print("\n--- Train Loss ---")
print("Train Loss:", checkpoint.get("train_loss", "N/A"))

# ================= TRAIN METRICS =================
print("\n--- Train Metrics ---")
train_metrics = checkpoint.get("train_metrics", {})

print("Accuracy:", train_metrics.get("acc", "N/A"))
print("Macro Precision:", train_metrics.get("macro_precision", "N/A"))
print("Macro Recall:", train_metrics.get("macro_recall", "N/A"))
print("Macro Sensitivity:", train_metrics.get("macro_sensitivity", "N/A"))
print("Macro Specificity:", train_metrics.get("macro_specificity", "N/A"))
print("Macro F1:", train_metrics.get("macro_f1", "N/A"))
print("QWK:", train_metrics.get("qwk", "N/A"))

# ================= VALIDATION LOSS =================
print("\n--- Validation Loss ---")
print("Validation Loss:", checkpoint.get("val_loss", "N/A"))

# ================= VALIDATION METRICS =================
print("\n--- Validation Metrics ---")
val_metrics = checkpoint.get("val_metrics", {})

print("Accuracy:", val_metrics.get("acc", "N/A"))
print("Macro Precision:", val_metrics.get("macro_precision", "N/A"))
print("Macro Recall:", val_metrics.get("macro_recall", "N/A"))
print("Macro Sensitivity:", val_metrics.get("macro_sensitivity", "N/A"))
print("Macro Specificity:", val_metrics.get("macro_specificity", "N/A"))
print("Macro F1:", val_metrics.get("macro_f1", "N/A"))
print("QWK:", val_metrics.get("qwk", "N/A"))

# ================= SETTINGS =================
print("\n--- Settings ---")
settings = checkpoint.get("settings", {})

for key, value in settings.items():
    print(f"{key}: {value}")

# ================= MODEL STATE INFO =================
model_state = checkpoint.get("model_state_dict", {})

print("\n--- Model State Dict ---")
print("Total tensors saved:", len(model_state))

# ================= ALL KEYS =================
print("\n--- Available Keys In Checkpoint ---")
for key in checkpoint.keys():
    print("-", key)

print("\nDone.")