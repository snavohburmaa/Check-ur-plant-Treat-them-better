"""Generate result charts from existing trained model."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.model_utils import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_model.pkl")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading model...", flush=True)
data = joblib.load(MODEL_PATH)
clf = data["classifier"]
le = data["label_encoder"]
classes = data["classes"]
acc = data["accuracy"]

print(f"  {len(classes)} classes, {acc:.1%} accuracy\n", flush=True)

print("Loading test data...", flush=True)
X, y = [], []
for cls_name in classes:
    cls_dir = os.path.join(DATASET_DIR, cls_name)
    if not os.path.isdir(cls_dir):
        continue
    images = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))][:15]
    for img_name in images:
        try:
            img = Image.open(os.path.join(cls_dir, img_name))
            X.append(extract_features(img))
            y.append(cls_name)
        except Exception:
            continue

X = np.array(X)
y_encoded = le.transform(y)
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
y_pred = clf.predict(X_test)

print("Generating charts...\n", flush=True)

# 1. Dataset distribution
df = pd.DataFrame({"class": y, "plant": [c.split("___")[0] for c in y]})
plant_counts = df["plant"].value_counts()
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(len(plant_counts)), plant_counts.values, color="#40916c")
ax.set_xticks(range(len(plant_counts)))
ax.set_xticklabels(plant_counts.index, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Number of Images")
ax.set_title("Dataset Distribution by Plant")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "dataset_distribution.png"), dpi=150)
plt.close()
print("  Saved: dataset_distribution.png", flush=True)

# 2. Accuracy per plant
plants = sorted(set(c.split("___")[0] for c in classes))
plant_acc = {}
for plant in plants:
    mask = np.array([classes[i].startswith(plant) for i in y_test])
    if mask.sum() > 0:
        plant_acc[plant] = accuracy_score(y_test[mask], y_pred[mask])

fig, ax = plt.subplots(figsize=(12, 5))
names = list(plant_acc.keys())
accs = list(plant_acc.values())
colors = ["#2d6a4f" if a >= 0.5 else "#d62828" for a in accs]
bars = ax.bar(range(len(names)), [a * 100 for a in accs], color=colors)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy per Plant")
ax.set_ylim(0, 100)
for bar, a in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{a:.0%}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_per_plant.png"), dpi=150)
plt.close()
print("  Saved: accuracy_per_plant.png", flush=True)

# 3. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(14, 12))
ax.imshow(cm, cmap="Greens", interpolation="nearest")
ax.set_title(f"Confusion Matrix ({len(classes)} classes)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("  Saved: confusion_matrix.png", flush=True)

# 4. Overall results
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis("off")
txt = (
    f"Plant Disease Classifier Results\n"
    f"{'=' * 40}\n\n"
    f"Total Classes:    {len(classes)}\n"
    f"Total Samples:    {len(y)}\n"
    f"Test Accuracy:    {acc:.1%}\n\n"
    f"Model:            Random Forest\n"
    f"Estimators:       50\n"
    f"Features:         HSV + Gradient + Color\n"
    f"Feature Size:     109"
)
ax.text(0.5, 0.5, txt, transform=ax.transAxes, fontsize=13,
        verticalalignment="center", horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#daf1e4", edgecolor="#2d6a4f"))
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "overall_results.png"), dpi=150)
plt.close()
print("  Saved: overall_results.png", flush=True)

# 5. Classification report CSV
report = pd.DataFrame(
    classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
).transpose()
report.to_csv(os.path.join(RESULTS_DIR, "classification_report.csv"))
print("  Saved: classification_report.csv", flush=True)

print(f"\nDone! All results in: {RESULTS_DIR}/", flush=True)
