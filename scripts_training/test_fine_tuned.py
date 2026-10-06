import os
import torch
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# ==========================================
# CHUNG KẾT: NGHIỆM THU DỰ ÁN DYSARTHRIA AI
# ==========================================

print("1. Đang tải Não gốc (Whisper-tiny)...")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="en", task="transcribe")
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

print("2. Đang cấy ghép Trí khôn mới (LoRA) vào não gốc...")
# Đường dẫn tới thư mục chứa 2 file tải từ Kaggle
lora_path = "LoRA_Dysarthria_Weights"

# Lắp ghép Adapter vào Base Model
model = PeftModel.from_pretrained(base_model, lora_path)

print("\n3. Bắt đầu nhận diện ca bệnh Dysarthria...")
audio_path = "data/real_patient.wav"

# Đọc file âm thanh bệnh nhân
audio_array, sr = sf.read(audio_path)

# Tiền xử lý âm thanh thành Spectrogram
input_features = processor(audio_array, sampling_rate=sr, return_tensors="pt").input_features

# Bắt đầu suy luận bằng bộ não mới
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

print("\n==================================")
print(f"🎯 ĐÁP ÁN ĐÚNG (Thực tế) : 'side'")
print(f"🤖 WHISPER CŨ (Lab trước): 'decide' (Sai bét)")
print(f"🚀 AI MỚI CỦA BẠN        : '{transcription}'")
print("==================================")
print("=> Nếu dòng cuối in ra chữ 'side' (hoặc gần giống), XIN CHÚC MỪNG, BẠN ĐÃ TẠO RA MỘT HỆ THỐNG AI Y TẾ THỰC THỤ!")
