import os
import time
import uuid
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

from model import load_model, predict_pil_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PTH_PATH = os.path.join(BASE_DIR, "pth", "deepfake2.pth")

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "deepfake_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model 1 lần khi app start (đúng rồi, đỡ tốn thời gian)
predictor = load_model(PTH_PATH)


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/predict")
def predict():
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "Chưa chọn ảnh"}), 400

    if not allowed_file(f.filename):
        return jsonify({"ok": False, "error": "Chỉ nhận png/jpg/jpeg/webp"}), 400

    # tránh đè file trùng tên: thêm uuid
    filename = secure_filename(f.filename)
    ext = filename.rsplit(".", 1)[1].lower()
    tmp_name = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_DIR, tmp_name)
    f.save(path)

    try:
        # --- BỎ WARNING PIL PALETTE TRANSPARENCY ---
        img = Image.open(path)
        # Nếu ảnh mode "P" (palette) có transparency -> convert RGBA trước sẽ hết warning
        if img.mode == "P":
            img = img.convert("RGBA")
        img = img.convert("RGB")

        # --- TIMING ---
        t0 = time.perf_counter()
        label_out, confidence, scores_by_class, probs_by_idx, idx_to_class = predict_pil_image(predictor, img)
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        real_score = scores_by_class.get("real")
        fake_score = scores_by_class.get("fake")

        return jsonify({
            "ok": True,
            "label": label_out,
            "confidence": round(float(confidence) * 100, 2),
            "real_score": round(float(real_score) * 100, 2) if real_score is not None else None,
            "fake_score": round(float(fake_score) * 100, 2) if fake_score is not None else None,
            "inference_ms": round(float(infer_ms), 2)  # ✅ thêm timing ở đây
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    finally:
        try:
            os.remove(path)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
