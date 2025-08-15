import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ===== 1. Load model và label encoder =====
model_path = r"E:\Machine Vision\XLA_01FIE\Dog_Pee_Pose_Estimation\police_sign_lstm_model.h5"
label_path = r"E:\Machine Vision\XLA_01FIE\Dog_Pee_Pose_Estimation\label_encoder.pkl"

model = load_model(model_path)

with open(label_path, 'rb') as f:
    label_encoder = pickle.load(f)

# ===== 2. Khởi tạo MediaPipe Pose =====
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# ===== 3. Hàm trích xuất landmark =====
def extract_landmarks(results):
    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(landmarks)
    else:
        return None

# ===== 4. Mở webcam và khởi tạo biến lưu trữ =====
cap = cv2.VideoCapture(0)
lm_list = []  # Danh sách lưu 50 frame landmarks
sequence_length = 50  # Số frame cần thu thập

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lật ảnh để giống gương
    frame = cv2.flip(frame, 1)

    # Chuyển sang RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dự đoán pose
    results = pose.process(image_rgb)

    # Vẽ skeleton
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Lấy landmark
    landmarks = extract_landmarks(results)
    if landmarks is not None:
        # Thêm landmarks vào danh sách
        lm_list.append(landmarks)

        # Kiểm tra nếu đã thu thập đủ 50 frame
        if len(lm_list) == sequence_length:
            # Chuyển lm_list thành numpy array và reshape cho LSTM
            input_data = np.array(lm_list).reshape(1, sequence_length, -1)

            # Dự đoán
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)
            label = label_encoder.inverse_transform([predicted_class])[0]

            # Hiển thị label
            cv2.putText(frame, f"Command: {label}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print(f"Command: {label}")

            # Xóa lm_list để bắt đầu chuỗi mới
            lm_list = []

    # Hiển thị số frame đã thu thập
    cv2.putText(frame, f"Frames: {len(lm_list)}/{sequence_length}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Hiển thị video
    cv2.imshow("Police Sign Recognition", frame)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()