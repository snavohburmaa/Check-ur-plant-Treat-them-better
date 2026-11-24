import json
import ssl
import base64
import urllib.request
import certifi

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


def get_disease_prompt(plant_name):
    known_diseases = []
    prefix = PLANT_NAME_MAP.get(plant_name, plant_name)
    for cls_name, info in DISEASE_DB.items():
        if cls_name.startswith(prefix):
            known_diseases.append(f"{info['disease']} (caused by {info['cause']})")
    diseases_list = "\n".join(f"- {d}" for d in known_diseases) if known_diseases else "Any known disease for this plant"

    return f"""You are a plant pathologist. Analyze this {plant_name} leaf photo.

Known {plant_name} diseases:
{diseases_list}

Return JSON with these fields (keep values short, 1-2 sentences max):
- disease: name or "Healthy"
- confidence: 0-100
- cause: brief cause
- symptoms: what you see
- severity: None/Moderate/High/Critical
- treatment: list of 3-4 short steps
- prevention: list of 2-3 short tips
- summary: one sentence"""


def parse_ai_result(content, plant_name):
    content = content.strip()
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                content = part
                break
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        content = content[start:end]
    parsed = json.loads(content)
    return {
        "plant": plant_name,
        "disease": parsed.get("disease", "Unknown"),
        "confidence": parsed.get("confidence", 80),
        "cause": parsed.get("cause", "Unknown"),
        "symptoms": parsed.get("symptoms", ""),
        "severity": parsed.get("severity", "Moderate"),
        "treatment": parsed.get("treatment", []),
        "prevention": parsed.get("prevention", []),
        "summary": parsed.get("summary", ""),
        "source": "ai",
    }


def diagnose_with_gemini(image_path, plant_name, api_key):
    if not api_key:
        return None

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")

    prompt = get_disease_prompt(plant_name)

    payload = json.dumps({
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": img_b64}},
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
        },
    }).encode()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
            result = json.loads(resp.read())
        content = result["candidates"][0]["content"]["parts"][0]["text"]
        return parse_ai_result(content, plant_name)
    except Exception as e:
        print(f"  Gemini API error: {e}", flush=True)
        return None
