import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ==========================================
# BÀI LAB 03: "ĐỨNG TRÊN VAI NGƯỜI KHỔNG LỒ" (HUGGING FACE & WHISPER)
# Thay vì huấn luyện AI từ đầu (mất hàng ngàn giờ chạy GPU),
# Ta sẽ mượn mô hình Whisper đã được OpenAI huấn luyện sẵn.
# ==========================================

print("1. Đang kết nối Hugging Face để tải Não bộ (Model) và Giác quan (Processor)...")
print("   (Lần chạy đầu tiên sẽ tốn chút thời gian để tải model về máy)")

# Processor: Bộ tiền xử lý (Thực chất nó tự động làm các việc giống hệt Bài Lab 01 của chúng ta)
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

# Model: Mạng Neural khổng lồ (bản tiny là bản nhẹ nhất để chạy test trên CPU)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None

print("2. Đang tải âm thanh thử nghiệm...")
# Tạm dùng âm thanh kèn trumpet (hoặc nếu bạn có file giọng nói thật, 
# hãy sửa 'trumpet' thành đường dẫn tới file của bạn, ví dụ: '../data/sample.wav')
y, sr = librosa.load(librosa.ex('trumpet'), sr=16000) 
# QUAN TRỌNG: Whisper bắt buộc âm thanh phải ép về chuẩn 16,000 Hz! 

print("3. Bắt đầu AI Nhận dạng (Inference)...")
# Đút âm thanh vào Processor để biến thành Ma Trận (Tensor)
input_features = processor(y, sampling_rate=sr, return_tensors="pt").input_features 

# Đưa Ma trận vào Não bộ Whisper để nó suy nghĩ (Tạo ra một chuỗi mã hóa Token)
predicted_ids = model.generate(input_features)

# Dịch ngược mã hóa Token đó thành Text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("\n==================================")
print(f"🎤 AI NGHE ĐƯỢC LÀ: {transcription}")
print("==================================")
print("(Ghi chú: Vì âm thanh nạp vào là tiếng kèn trumpet không có lời, nên AI sẽ cố 'tưởng tượng' ra một câu gì đó. Hãy thử tự ghi âm giọng của bạn rồi sửa đường dẫn load file nhé!)")
