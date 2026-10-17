import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# 全域變數延遲初始化，避免 Gunicorn 啟動時直接死鎖
hands = None

def get_hands_instance():
    global hands
    if hands is None:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    return hands

def count_raised_fingers(hand_landmarks):
    landmarks = hand_landmarks.landmark
    fingers = [
        1 if landmarks[8].y < landmarks[6].y else 0,   # 食指
        1 if landmarks[12].y < landmarks[10].y else 0, # 中指
        1 if landmarks[16].y < landmarks[14].y else 0, # 無名指
        1 if landmarks[20].y < landmarks[18].y else 0  # 小指
    ]
    return sum(fingers)

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "running", "message": "ESP32-S3-CAM Gesture Recognition API is active!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        img_bytes = file.read()

        if not img_bytes:
            return jsonify({'error': 'Empty image file'}), 400

        # 轉為 OpenCV 影像格式
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # 💡 關鍵：嚴格強制縮放到極小解析度 (240px)，避免 MediaPipe 計算量過大卡死
        height, width = img.shape[:2]
        if width > 240:
            new_height = int(height * (240 / width))
            img = cv2.resize(img, (240, new_height), interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 動態取得 MediaPipe 實例
        detector = get_hands_instance()
        results = detector.process(img_rgb)

        command = "STOP"

        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                raised_count = count_raised_fingers(hand_landmarks)

                if raised_count >= 3:
                    command = "FORWARD"
                elif 1 <= raised_count <= 2:
                    command = "BACKWARD"
                else:
                    command = "STOP"

        return jsonify({'command': command})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)