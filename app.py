from flask import Flask, request, jsonify
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("best.pt")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.5)
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                classes = boxes.cls.tolist()
                frames_data.append(classes)

    cap.release()
    return frames_data

@app.route("/predict", methods=["POST"])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files['video']
    video_path = "temp.mp4"
    video_file.save(video_path)

    try:
        data = process_video(video_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(video_path)

    return jsonify({"detections": data})
