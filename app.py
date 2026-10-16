import cv2
import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp

# 相容性較佳的 MediaPipe 載入方式 (修復 AttributeError: module 'mediapipe' has no attribute 'solutions')
try:
    from mediapipe.python.solutions import hands as mp_hands
except ImportError:
    try:
        mp_hands = mp.solutions.hands
    except AttributeError:
        import mediapipe.solutions.hands as mp_hands

app = Flask(__name__)

# 初始化手勢辨識模型
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def count_raised_fingers(hand_landmarks):
    """
    計算伸直的手指數量：
    - 食指 (8)、中指 (12)、無名指 (16)、小指 (20) 的 Tip 點 Y 座標小於 PIP 點 Y 座標代表伸直
    - 大拇指 (4) 的 Tip 點 X 座標比較 (簡易判斷)
    """
    landmarks = hand_landmarks.landmark
    fingers = []

    # 1. 食指 (Index)
    fingers.append(1 if landmarks[8].y < landmarks[6].y else 0)
    # 2. 中指 (Middle)
    fingers.append(1 if landmarks[12].y < landmarks[10].y else 0)
    # 3. 無名指 (Ring)
    fingers.append(1 if landmarks[16].y < landmarks[14].y else 0)
    # 4. 小指 (Pinky)
    fingers.append(1 if landmarks[20].y < landmarks[18].y else 0)

    total_raised = sum(fingers)
    return total_raised

@app.route('/', methods=['GET'])
def index():
    """根目錄測試，確認 API 運作中"""
    return jsonify({"status": "running", "message": "ESP32-S3-CAM Gesture Recognition API is active!"})

@app.route('/predict', methods=['POST'])
def predict():
    """接收 ESP32 傳送的影像並判斷手勢"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()
    
    if not img_bytes:
        return jsonify({'error': 'Empty image file'}), 400

    # 將位元組陣列解碼為 OpenCV 影像格式
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    # 轉為 RGB 供 MediaPipe 處理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    command = "STOP"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            raised_count = count_raised_fingers(hand_landmarks)

            # 手勢控制邏輯：
            # - 伸出 3 根以上手指（如張開手掌）：前進 FORWARD
            # - 伸出 1~2 根手指（如比 1 或 2）：後退 BACKWARD
            # - 握拳 (0 根）：停止 STOP
            if raised_count >= 3:
                command = "FORWARD"
            elif 1 <= raised_count <= 2:
                command = "BACKWARD"
            else:
                command = "STOP"
    else:
        # 未偵測到手部預設停止
        command = "STOP"

    return jsonify({'command': command})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)