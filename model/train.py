"""Train Plant Disease Classifier with result visualization."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.model_utils import extract_features

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_model.pkl")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MAX_IMAGES = 15


def load_dataset():
    X, y = [], []

    classes = sorted([d for d in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, d))
                      and not d.startswith(".")])

    print(f"  Found {len(classes)} classes", flush=True)

    for cls_name in classes:
        cls_dir = os.path.join(DATASET_DIR, cls_name)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        images = images[:MAX_IMAGES]

        count = 0
        for img_name in images:
            path = os.path.join(cls_dir, img_name)
            try:
                img = Image.open(path)
                feat = extract_features(img)
                X.append(feat)
                y.append(cls_name)
                count += 1
            except Exception:
                continue

        print(f"  [{count:>4} images] {cls_name}", flush=True)

    return np.array(X), np.array(y), classes


def plot_dataset_distribution(df, save_path):
    plant_counts = df["plant"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(plant_counts)), plant_counts.values, color="#40916c")
    ax.set_xticks(range(len(plant_counts)))
    ax.set_xticklabels(plant_counts.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Images")
    ax.set_title("Dataset Distribution by Plant")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


def plot_accuracy_per_plant(y_true, y_pred, le, save_path):
    classes = le.classes_
    plants = sorted(set(c.split("___")[0] for c in classes))

    plant_acc = {}
    for plant in plants:
        mask = np.array([classes[i].startswith(plant) for i in y_true])
        if mask.sum() > 0:
            plant_acc[plant] = accuracy_score(y_true[mask], y_pred[mask])

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
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


def plot_confusion_overview(y_true, y_pred, le, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Greens", interpolation="nearest")
    ax.set_title(f"Confusion Matrix ({len(le.classes_)} classes)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


def plot_overall_results(acc, num_classes, num_samples, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    results_text = (
        f"Plant Disease Classifier Results\n"
        f"{'=' * 40}\n\n"
        f"Total Classes:    {num_classes}\n"
        f"Total Samples:    {num_samples}\n"
        f"Test Accuracy:    {acc:.1%}\n\n"
        f"Model:            Random Forest\n"
        f"Estimators:       100\n"
        f"Max Depth:        20\n"
        f"Features:         HSV + Gradient + Color\n"
        f"Feature Size:     109"
    )
    ax.text(0.5, 0.5, results_text, transform=ax.transAxes,
            fontsize=13, verticalalignment="center", horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#daf1e4", edgecolor="#2d6a4f"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


def train():
    print("\n=== Plant Disease Classifier Training ===\n", flush=True)

    if not os.path.exists(DATASET_DIR):
        print("Dataset not found. Place dataset in /dataset folder.", flush=True)
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading dataset and extracting features...", flush=True)
    X, y, classes = load_dataset()
    print(f"\n  Total: {len(X)} samples, {len(classes)} classes", flush=True)
    print(f"  Feature vector size: {X.shape[1]}", flush=True)

    df = pd.DataFrame({"class": y, "plant": [c.split("___")[0] for c in y]})
    print(f"\n  Dataset summary:\n{df['plant'].value_counts().to_string()}\n", flush=True)

    print("Generating dataset distribution chart...", flush=True)
    plot_dataset_distribution(df, os.path.join(RESULTS_DIR, "dataset_distribution.png"))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("\nSplitting train/test (80/20)...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}", flush=True)

    print("\nTraining Random Forest...", flush=True)
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    print("\nEvaluating...", flush=True)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test accuracy: {acc:.1%}\n", flush=True)

    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    print(report, flush=True)

    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    ).transpose()
    report_df.to_csv(os.path.join(RESULTS_DIR, "classification_report.csv"))
    print(f"  Saved: classification_report.csv", flush=True)

    print("\nGenerating result charts...", flush=True)
    plot_accuracy_per_plant(y_test, y_pred, le, os.path.join(RESULTS_DIR, "accuracy_per_plant.png"))
    plot_confusion_overview(y_test, y_pred, le, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_overall_results(acc, len(classes), len(X), os.path.join(RESULTS_DIR, "overall_results.png"))

    model_data = {
        "classifier": clf,
        "label_encoder": le,
        "classes": classes,
        "feature_size": X.shape[1],
        "accuracy": acc,
    }
    joblib.dump(model_data, MODEL_PATH, compress=3)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n  Model saved to: {MODEL_PATH} ({size_mb:.1f} MB)", flush=True)

    print(f"\n=== Done! Accuracy: {acc:.1%} on {len(classes)} classes ===", flush=True)
    print(f"  Results saved in: {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    train()
