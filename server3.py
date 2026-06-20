from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
import pickle
from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ==================================================
# LOAD MODELS ONCE (IMPORTANT)
# ==================================================

# YOLO for object detection
print("[SERVER] Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")

# FaceNet embedder
print("[SERVER] Loading FaceNet model...")
embedder = FaceNet()

# Load FaceNet embeddings
print("[SERVER] Loading FaceNet embeddings...")
with open("facenet_embeddings.pickle", "rb") as f:
    data = pickle.load(f)

known_embeddings = np.array(data["embeddings"])
known_names = data["names"]

print("[SERVER] Models loaded successfully")


# ==================================================
# MAIN PROCESS ROUTE
# ==================================================
@app.route("/process", methods=["POST"])
def process():

    if "image" not in request.files:
        return jsonify({"error": "No image received"}), 400

    mode = request.form.get("mode")
    img_bytes = request.files["image"].read()

    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    print("\n[SERVER] Request received | Mode:", mode)

    # ==================================================
    # 📖 READING MODE (OCR)
    # ==================================================
    if mode == "reading":

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (800, 600), interpolation=cv2.INTER_AREA)

        text = pytesseract.image_to_string(rgb).strip()

        print("[SERVER] OCR RESULT:")
        print(text if text else "[NO TEXT DETECTED]")

        return jsonify({
            "mode": "reading",
            "text": text
        })

    # ==================================================
    # 🎯 OBJECT IDENTIFICATION MODE
    # ==================================================
    elif mode == "object":

        results = yolo_model(img)
        detected_objects = set()

        for r in results:
            for cls in r.boxes.cls:
                detected_objects.add(yolo_model.names[int(cls)])

        detected_objects = list(detected_objects)

        print("[SERVER] Objects detected:", detected_objects if detected_objects else "[NONE]")

        return jsonify({
            "mode": "object",
            "objects": detected_objects
        })

    # ==================================================
    # 🙂 FACE RECOGNITION MODE (FACENET)
    # ==================================================
    elif mode == "face":

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = embedder.extract(rgb, threshold=0.95)
        recognized_faces = []

        if not faces:
            print("[SERVER] No face detected")

        for face in faces:
            embedding = face["embedding"].reshape(1, -1)

            similarities = cosine_similarity(embedding, known_embeddings)[0]
            best_index = np.argmax(similarities)
            best_score = similarities[best_index]

            # Threshold (tunable)
            if best_score > 0.55:
                name = known_names[best_index]
            else:
                name = "Unknown"

            recognized_faces.append(name)

        print("[SERVER] Faces recognized:", recognized_faces if recognized_faces else "[NONE]")

        return jsonify({
            "mode": "face",
            "faces": recognized_faces
        })

    # ==================================================
    # INVALID MODE
    # ==================================================
    else:
        return jsonify({"error": "Invalid mode"}), 400


# ==================================================
# START SERVER
# ==================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
