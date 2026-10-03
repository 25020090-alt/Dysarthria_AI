import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

print("Đang tải dữ liệu âm thanh mẫu...")
# Tải một file âm thanh kèn trumpet có sẵn trong librosa để làm vật liệu test
y, sr = librosa.load(librosa.ex('trumpet'))

print(f"1. Tín hiệu gốc (Vector 1 chiều):")
print(f" - Kích thước vector y: {y.shape}")
print(f" - Tần số lấy mẫu (Sample Rate): {sr} Hz (nghĩa là 1 giây có {sr} con số)")
print(f" - Độ dài đoạn ghi âm: {len(y)/sr:.2f} giây\n")

# Chuyển đổi Vector thành Ma trận đặc trưng phổ Mel (Mel-spectrogram)
# Đây chính là định dạng "hình ảnh" mà mô hình AI (như Whisper) sẽ nhìn vào
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)

print(f"2. Ma trận đặc trưng (Mel-spectrogram):")
print(f" - Kích thước ma trận S_dB: {S_dB.shape}")
print(f"   (Bao gồm 128 dải tần số x {S_dB.shape[1]} khung thời gian)")
print(f" - Kiểu dữ liệu: {type(S_dB)}\n")

# Vẽ ma trận ra màn hình
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram (Ma trận biểu diễn âm thanh)')
plt.tight_layout()

# Thay vì plt.show() (có thể làm treo terminal), mình sẽ lưu thành file ảnh để bạn dễ dàng xem lại
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trumpet_spectrogram.png')
plt.savefig(output_path)
print(f" Đã lưu hình ảnh Spectrogram tại: {output_path}")

plt.show() # Vẫn giữ lại plt.show() để hiển thị cửa sổ nếu bạn chạy trên VS Code
