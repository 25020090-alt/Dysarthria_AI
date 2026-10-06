import os
import soundfile as sf
import io
from datasets import load_dataset, Audio
from faster_whisper import WhisperModel
import ollama

# ==========================================
# BÀI TEST THỰC TẾ: ĐỐI ĐẦU VỚI GIỌNG DYSARTHRIA
# ==========================================

print("\n1. Đang tải Dataset từ Hugging Face (Ankesh1234/torgo-dysarthria-male-words-100)...")
# Tải tập dữ liệu. Dùng decode=False để tránh lỗi thiếu thư viện giải mã âm thanh của Windows (torchcodec)
dataset = load_dataset("Ankesh1234/torgo-dysarthria-male-words-100", split="train")
dataset = dataset.cast_column("audio", Audio(decode=False))

# Chọn ngẫu nhiên bệnh nhân số 0 trong tập dữ liệu
sample = dataset[0]
ground_truth = sample["transcription"] # Đây là đáp án chuẩn!
audio_bytes = sample["audio"]["bytes"] # Lấy dữ liệu file gốc dưới dạng byte

print("\n===============================")
print(f"🎯 ĐÁP ÁN ĐÚNG (Bệnh nhân đang muốn nói chữ gì): '{ground_truth}'")
print("===============================")

# Dùng soundfile giải mã byte âm thanh và lưu ra ổ cứng
audio_array, sr = sf.read(io.BytesIO(audio_bytes))
audio_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real_patient.wav')
sf.write(audio_path, audio_array, sr)

print("\n2. Đang dùng Faster-Whisper (Chưa huấn luyện) để nghe thử...")
audio_model = WhisperModel("tiny", device="cpu", compute_type="int8")
segments, _ = audio_model.transcribe(audio_path, beam_size=5)
whisper_text = " ".join([s.text for s in segments]).strip()

print(f"=> 🤖 WHISPER NGHE ĐƯỢC LÀ: '{whisper_text}'")

print("\n3. Đang nhờ Ollama (Qwen) sửa lỗi đoạn văn của Whisper...")
final_text = "Lỗi Ollama"
try:
    response = ollama.chat(
        model='qwen2.5:0.5b', 
        messages=[
            # Vì bộ TORGO là tiếng Anh, nên ta yêu cầu LLM sửa tiếng Anh
            {'role': 'system', 'content': 'You are a medical assistant. Fix any spelling/phonetic errors from a dysarthric speaker. Output ONLY the corrected English word without explanations.'},
            {'role': 'user', 'content': whisper_text}
        ]
    )
    final_text = response['message']['content']
    print(f"=> 👨‍⚕️ OLLAMA SỬA THÀNH: '{final_text}'")
except Exception as e:
    print("Ollama chưa bật hoặc lỗi kết nối.")

print("\n===============================")
print("KẾT LUẬN CUỐI CÙNG:")
print(f"Bệnh nhân nói  : {ground_truth}")
print(f"Whisper dịch   : {whisper_text}")
print(f"Ollama sửa lại : {final_text}")
print("===============================")
print("=> Nếu Whisper và Ollama đều không khớp với đáp án đúng, chúc mừng bạn! Đây chính là lý do bạn CẦN huấn luyện dự án này.")
print("=> Bạn sẽ dùng bộ data này để Fine-tune lại não bộ Whisper. Khi Fine-tune xong, Whisper dịch sẽ khớp 100% với đáp án!")
