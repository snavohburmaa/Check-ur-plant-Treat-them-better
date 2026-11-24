import os
import numpy as np
from PIL import Image
import joblib

from utils.disease_info import DISEASE_DB


PLANT_NAME_MAP = {
    "Apple": "Apple",
    "Blueberry": "Blueberry",
    "Cherry": "Cherry_(including_sour)",
    "Corn": "Corn_(maize)",
    "Grape": "Grape",
    "Orange": "Orange",
    "Peach": "Peach",
    "Bell Pepper": "Pepper,_bell",
    "Potato": "Potato",
    "Raspberry": "Raspberry",
    "Soybean": "Soybean",
    "Squash": "Squash",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
}


def extract_features(pil_img):
    img = pil_img.resize((64, 64)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    arr_norm = arr / 255.0
    r_n, g_n, b_n = arr_norm[:, :, 0], arr_norm[:, :, 1], arr_norm[:, :, 2]
    cmax = np.max(arr_norm, axis=2)
    cmin = np.min(arr_norm, axis=2)
    delta = cmax - cmin + 1e-7

    h = np.zeros_like(cmax)
    mask_r = cmax == r_n
    mask_g = (cmax == g_n) & ~mask_r
    mask_b = ~mask_r & ~mask_g
    h[mask_r] = 60 * (((g_n[mask_r] - b_n[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60 * (((b_n[mask_g] - r_n[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60 * (((r_n[mask_b] - g_n[mask_b]) / delta[mask_b]) + 4)

    s = np.where(cmax == 0, 0, delta / (cmax + 1e-7))
    v = cmax

    h_scaled = (h / 360 * 255).astype(np.float32)
    s_scaled = (s * 255).astype(np.float32)
    v_scaled = (v * 255).astype(np.float32)

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    features = []

    for ch in [h_scaled, s_scaled, v_scaled]:
        hist, _ = np.histogram(ch, bins=32, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
        features.extend(hist.tolist())

    for ch in [h_scaled, s_scaled, v_scaled, gray]:
        features.append(float(ch.mean()))
        features.append(float(ch.std()))

    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    mag = np.sqrt(gx**2 + gy**2)
    features.append(float(mag.mean()))
    features.append(float(mag.std()))

    total = r + g + b + 1e-7
    features.append(float((g / total).mean()))
    features.append(float((r / total).mean()))
    features.append(float((b / total).mean()))

    return np.array(features, dtype=np.float32)


def load_model(model_path):
    if not os.path.exists(model_path):
        print("  Model not found (will use API only)", flush=True)
        return None
    try:
        data = joblib.load(model_path)
        print(f"  Model loaded: {len(data['classes'])} classes, {data['accuracy']:.1%} accuracy", flush=True)
        return data
    except Exception as e:
        print(f"  Model failed to load: {e}", flush=True)
        return None


def diagnose_with_model(pil_img, plant_name, model_data):
    if model_data is None:
        return None

    feat = extract_features(pil_img).reshape(1, -1)
    clf = model_data["classifier"]
    le = model_data["label_encoder"]

    proba = clf.predict_proba(feat)[0]

    if plant_name and plant_name in PLANT_NAME_MAP:
        prefix = PLANT_NAME_MAP[plant_name]
        mask = np.array([cls.startswith(prefix) for cls in le.classes_])
        filtered = proba.copy()
        filtered[~mask] = 0
        total = filtered.sum()
        if total > 0:
            proba = filtered / total

    top_indices = proba.argsort()[::-1][:3]
    results = []
    for idx in top_indices:
        if proba[idx] <= 0:
            continue
        cls_name = le.inverse_transform([idx])[0]
        confidence = int(proba[idx] * 100)
        info = DISEASE_DB.get(cls_name, {})
        results.append({
            "plant": info.get("plant", cls_name.split("___")[0]),
            "disease": info.get("disease", cls_name.split("___")[-1].replace("_", " ")),
            "confidence": confidence,
            "cause": info.get("cause", "Unknown"),
            "symptoms": info.get("symptoms", ""),
            "treatment": info.get("treatment", []),
            "severity": info.get("severity", "Unknown"),
            "source": "model",
        })
    return results
