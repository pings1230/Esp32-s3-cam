import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6
)

def detect_gesture(landmarks):
    # 手指關鍵點判定
    index_up = landmarks[8].y < landmarks[6].y
    index_down = landmarks[8].y > landmarks[6].y

    middle_up = landmarks[12].y < landmarks[10].y
    middle_down = landmarks[12].y > landmarks[10].y

    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y

    # 1. 張開五隻手指 -> 停止
    if index_up and middle_up and ring_up and pinky_up:
        return "STOP"

    # 2. 食指比上面 -> 正轉
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "FORWARD"

    # 3. 食指和中指同時比下面 -> 反轉
    if index_down and middle_down and not ring_up and not pinky_up:
        return "REVERSE"

    return "NONE"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    file = request.files['image']
    image_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"status": "error", "message": "Invalid image format"}), 400

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    command = "NONE"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            command = detect_gesture(hand_landmarks.landmark)

    return jsonify({"command": command, "status": "success"})

@app.route('/ping', methods=['GET'])
def ping():
    return "OK", 200

if __name__ == '__main__':
    # 讀取雲端環境指定的 PORT，預設為 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)