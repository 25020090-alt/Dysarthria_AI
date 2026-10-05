import os
import torch
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# ==========================================
# BÀI KIỂM TRA CUỐI CÙNG VỚI NÃO BỘ BIG DATA
# ==========================================

# CẢNH BÁO QUAN TRỌNG: 
# Lần này chúng ta phải dùng "whisper-small" làm não gốc (thay vì tiny)
# Nếu dùng sai Não gốc, cục LoRA sẽ lắp không vừa và báo lỗi ngay lập tức!
print("1. Đang tải Não gốc (Whisper-small)... (File gốc khá nặng, vui lòng đợi)")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

print("\n2. Đang cấy ghép Trí khôn Big Data (LoRA 5000 steps)...")
lora_path = "LoRA_BigData_Weights"

if not os.path.exists(lora_path) or not os.listdir(lora_path):
    print(f"\n[LỖI]: Bạn chưa kéo 2 file từ Kaggle vào thư mục {lora_path} trên VS Code!")
    exit()

# Lắp ghép Adapter vào Base Model
model = PeftModel.from_pretrained(base_model, lora_path)

print("\n3. Bắt đầu nhận diện ca bệnh Dysarthria với não bộ mới...")
audio_path = "data/real_patient.wav"

if not os.path.exists(audio_path):
    print(f"\n[LỖI]: Không tìm thấy file âm thanh {audio_path}")
    exit()

audio_array, sr = sf.read(audio_path)
input_features = processor(audio_array, sampling_rate=sr, return_tensors="pt").input_features

predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

print("\n==================================")
print(f"🎯 ĐÁP ÁN ĐÚNG (Thực tế) : 'side'")
print(f"🚀 AI BIG DATA (8GB)     : '{transcription}'")
print("==================================")
