from faster_whisper import WhisperModel
import librosa
import soundfile as sf
import os

# ==========================================
# BÀI LAB 05: HỆ THỐNG OFFLINE TỐC ĐỘ CAO (FASTER-WHISPER)
# 
# Vấn đề: Code HuggingFace (Lab 03) chạy trên CPU rất chậm vì kiến trúc của nó 
# được tối ưu cho Card đồ họa đắt tiền (GPU). Ở bệnh viện hoặc phòng khám, 
# bác sĩ chỉ có laptop văn phòng (CPU). Nếu phải đợi 10 giây mới dịch xong 1 câu 
# thì không thể giao tiếp thực tế được.
#
# Giải pháp: Faster-Whisper. 
# Công cụ này viết lại toàn bộ khung xương của AI bằng ngôn ngữ C/C++ 
# và "nén" các con số (quantization) về chuẩn INT8 siêu nhẹ.
# Kết quả: Nhanh gấp 4 lần, tốn ít RAM hơn 2 lần trên máy tính thường!
# ==========================================

print("1. Đang khởi động Engine C++ siêu tốc (Faster-Whisper)...")
# Khởi tạo mô hình bản 'tiny' (Nó sẽ tự tải model nén về nếu chưa có)
# device="cpu": Chạy thẳng trên chip máy tính văn phòng.
# compute_type="int8": Kỹ thuật nén não bộ AI xuống mức 8-bit cực nhẹ.
model = WhisperModel("tiny", device="cpu", compute_type="int8")

print("\n2. Đang chuẩn bị file âm thanh...")
# Mình tạm xuất file âm thanh kèn trumpet ra ổ cứng để Faster-Whisper đọc.
# (Thực tế sau này bạn chỉ cần vứt đường dẫn 'data/giong_noi.wav' vào là xong)
y, sr = librosa.load(librosa.ex('trumpet'), sr=16000)
temp_audio_path = "trumpet_temp.wav"
sf.write(temp_audio_path, y, sr)

print("\n3. Bắt đầu nhận diện (Inference)...")
print("=> Hãy để ý tốc độ phản hồi từ lúc dòng này hiện ra nhé!\n")

# Hàm transcribe tự động làm mọi thứ: Cắt Audio -> Tạo Spectrogram -> Gọi AI
segments, info = model.transcribe(temp_audio_path, beam_size=5)

print(f"=> Ngôn ngữ AI đoán được: '{info.language}' (Độ tự tin: {info.language_probability*100:.1f}%)")

print("\n==================================")
print("🎤 KẾT QUẢ AI NGHE ĐƯỢC (HIỂN THỊ THEO THỜI GIAN THỰC):")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text}")
print("==================================")

# Dọn dẹp file tạm
if os.path.exists(temp_audio_path):
    os.remove(temp_audio_path)
    
print("\n=> KẾT LUẬN: Code này sạch hơn, ngốn ít RAM hơn và tốc độ dịch thì nhanh hơn hẳn so với Lab 03!")
