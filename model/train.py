"""Train Plant Disease Classifier using Pillow"""

import os
import sys
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.model_utils import extract_features

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_model.pkl")
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


def train():
    print("\n=== Plant Disease Classifier Training ===\n", flush=True)

    if not os.path.exists(DATASET_DIR):
        print("Dataset not found. Place dataset in /dataset folder.", flush=True)
        return

    print("Loading dataset and extracting features...", flush=True)
    X, y, classes = load_dataset()
    print(f"\n  Total: {len(X)} samples, {len(classes)} classes", flush=True)
    print(f"  Feature vector size: {X.shape[1]}", flush=True)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("\nSplitting train/test (80/20)...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}", flush=True)

    print("\nTraining Random Forest...", flush=True)
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    print("\nEvaluating...", flush=True)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test accuracy: {acc:.1%}\n", flush=True)

    model_data = {
        "classifier": clf,
        "label_encoder": le,
        "classes": classes,
        "feature_size": X.shape[1],
        "accuracy": acc,
    }
    joblib.dump(model_data, MODEL_PATH, compress=3)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"  Saved to: {MODEL_PATH} ({size_mb:.1f} MB)", flush=True)
    print(f"\n=== Done! Accuracy: {acc:.1%} on {len(classes)} classes ===\n", flush=True)


if __name__ == "__main__":
    train()
