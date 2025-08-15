import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Đường dẫn đến thư mục chứa các file CSV
data_dir = r"E:\Machine Vision\XLA_01FIE\Dog_Pee_Pose_Estimation"
labels = ["ALL_STOP", "Front_and_Back_STOP", "Left_Faster", "Right_Faster", "Left_STOP", "Right_STOP", "NO_COMMAND"]
csv_files = [os.path.join(data_dir, f"{label}.txt") for label in labels]

# Đọc và xử lý dữ liệu
def load_data():
    X, y = [], []
    for label, csv_file in zip(labels, csv_files):
        if not os.path.exists(csv_file):
            print(f"File {csv_file} không tồn tại!")
            continue
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            X.append(row.values)  # Lấy toàn bộ cột (landmark features)
            y.append(label)
    return np.array(X), np.array(y)

# Tải dữ liệu
X, y = load_data()

# Mã hóa nhãn
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Reshape dữ liệu cho LSTM: [samples, timesteps, features]
# Giả sử mỗi frame là 1 timestep, và số cột trong CSV là số features
n_features = X_train.shape[1]
X_train = X_train.reshape((X_train.shape[0], 1, n_features))
X_test = X_test.reshape((X_test.shape[0], 1, n_features))

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(1, n_features), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# In tóm tắt mô hình
model.summary()

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# In báo cáo phân loại
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=labels))

# Vẽ biểu đồ độ chính xác và mất mát
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Lưu mô hình
model.save(os.path.join(data_dir, "police_sign_lstm_model.h5"))
print(f"Mô hình đã được lưu vào: {os.path.join(data_dir, 'police_sign_lstm_model.h5')}")

# Lưu label encoder để sử dụng sau này
import pickle
with open(os.path.join(data_dir, "label_encoder.pkl"), 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder đã được lưu vào: {os.path.join(data_dir, 'label_encoder.pkl')}")