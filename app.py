from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import onnxruntime as ort

app = Flask(__name__)
model_path = "best.onnx"

# تحميل الموديل مرة واحدة
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Unable to open video file")

    detections = []
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        try:
            outputs = session.run(None, {input_name: img})[0]
            detections_frame = []
            for det in np.transpose(outputs[0], (1, 0)):
                x_center, y_center, width, height, conf = det
                if conf < 0.5:
                    continue
                x1 = int((x_center - width / 2) * frame_w / 640)
                y1 = int((y_center - height / 2) * frame_h / 640)
                x2 = int((x_center + width / 2) * frame_w / 640)
                y2 = int((y_center + height / 2) * frame_h / 640)
                detections_frame.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": round(float(conf), 3)
                })
            detections.append(detections_frame)
        except Exception as e:
            print("Error:", e)
            continue

    cap.release()
    return detections

@app.route("/predict", methods=["POST"])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files['video']
    video_path = "temp_input.mp4"

    try:
        video_file.save(video_path)
        result = process_video(video_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

    return jsonify({"detections": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
