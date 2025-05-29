from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import onnxruntime as ort

app = Flask(__name__)
model_path = "best.onnx"
session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

def preprocess(frame):
    resized = cv2.resize(frame, (640, 640))  # حسب أبعاد الموديل
    img = resized.transpose(2, 0, 1)  # HWC → CHW
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_classes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})

        pred = outputs[0]
        if pred.shape[1] > 0:
            class_ids = pred[0][:, 5].astype(int).tolist()
            all_classes.append(class_ids)

    cap.release()
    return all_classes

@app.route("/predict", methods=["POST"])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files['video']
    video_path = "temp_video.mp4"
    video_file.save(video_path)

    try:
        result = process_video(video_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(video_path)

    return jsonify({"detections": result})