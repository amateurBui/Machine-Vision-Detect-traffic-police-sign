import cv2
import mediapipe as mp
import pandas as pd
import os

# Đường dẫn để lưu file CSV
output_dir = r"E:\Machine Vision\XLA_01FIE\Dog_Pee_Pose_Estimation"
os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
output_file = os.path.join(output_dir, "NO_COMMAND.txt")

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam!")
    exit()

# Khởi tạo thư viện Mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

lm_list = []  # Danh sách lưu các landmark
label = "NO_COMMAND"  # Nhãn cho tư thế hiệu lệnh CSGT
no_of_frames = 600  # Số frame tối đa để thu thập

def make_landmark_timestep(results):
    """Chuyển đổi các landmark thành danh sách tọa độ (x, y, z, visibility)."""
    if not results.pose_landmarks:
        return None
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    """Vẽ các landmark và kết nối lên ảnh."""
    if results.pose_landmarks:
        # Vẽ các đường nối
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Vẽ các điểm landmark
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)  # Đổi màu thành xanh lá để dễ nhận diện
    return img

# Thu thập dữ liệu
print(f"Bắt đầu thu thập dữ liệu pose cho tư thế {label}...")
while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame!")
        break

    # Nhận diện pose
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)

    if results.pose_landmarks:
        # Ghi nhận thông số khung xương
        lm = make_landmark_timestep(results)
        if lm:
            lm_list.append(lm)
        # Vẽ khung xương lên ảnh
        frame = draw_landmark_on_image(mpDraw, results, frame)

    # Hiển thị ảnh
    cv2.imshow("Pose Estimation - Press 'q' to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Hiển thị tiến độ
    print(f"Đã thu thập {len(lm_list)}/{no_of_frames} frames...", end='\r')

# Lưu vào file CSV
if lm_list:
    df = pd.DataFrame(lm_list)
    df.to_csv(output_file, index=False)
    print(f"\nDữ liệu đã được lưu vào: {output_file}")
else:
    print("\nKhông thu thập được dữ liệu pose. Vui lòng kiểm tra webcam hoặc tư thế!")

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()