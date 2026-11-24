import os
from PIL import Image
from flask import Flask, render_template, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from utils._engine import diagnose_with_gemini
from utils.model_utils import load_model, diagnose_with_model

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_model.pkl")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ENV_PATH = os.path.join(BASE_DIR, ".venv", ".env")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

model_data = None
GEMINI_API_KEY = None


def load_env():
    global GEMINI_API_KEY
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if key == "GEMINI_API_KEY":
                        GEMINI_API_KEY = val
    if GEMINI_API_KEY:
        print("  Gemini API key loaded (primary)", flush=True)
    else:
        print("  WARNING: No Gemini API key, using model only", flush=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/diagnose", methods=["POST"])
def diagnose():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file uploaded"})

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"ok": False, "error": "No file selected"})

    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"ok": False, "error": "Only PNG, JPG, WEBP allowed"})

    selected_plant = request.form.get("plant", "").strip()
    if not selected_plant:
        return jsonify({"ok": False, "error": "Please select a plant first"})

    filename = secure_filename(f.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    f.save(filepath)

    try:
        pil_img = Image.open(filepath)
    except Exception:
        os.remove(filepath)
        return jsonify({"ok": False, "error": "Cannot read image"})

    print(f"  Diagnosing: plant={selected_plant}, file={filename}", flush=True)

    result = diagnose_with_gemini(filepath, selected_plant, GEMINI_API_KEY)
    if result:
        print(f"  Gemini result: {result['disease']}", flush=True)
        return jsonify({"ok": True, "results": [result]})

    print("  Gemini failed, trying model...", flush=True)
    model_results = diagnose_with_model(pil_img, selected_plant, model_data)
    if model_results:
        print(f"  Model result: {model_results[0]['disease']}", flush=True)
        return jsonify({"ok": True, "results": model_results})

    print("  Both methods failed!", flush=True)
    return jsonify({"ok": False, "error": "Diagnosis failed. Please try again."})


@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    print("\n  Plant Doctor - Plant Disease Detector")
    print("  Open http://localhost:8080 in your browser\n", flush=True)
    load_env()
    model_data = load_model(MODEL_PATH)
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
