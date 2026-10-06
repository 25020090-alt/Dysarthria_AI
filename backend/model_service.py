import torch
import io
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

class DysarthriaModelService:
    def __init__(self):
        print("[INFO] Đang khởi động AI Dysarthria...")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")
        base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        
        lora_path = "ai_models/lora_big_data"
        try:
            self.model = PeftModel.from_pretrained(base_model, lora_path)
            print("[INFO] Đã nạp thành công bộ não Big Data 8GB!")
        except Exception as e:
            print(f"[CẢNH BÁO] Không tìm thấy LoRA tại {lora_path}, đang chạy tạm bằng Base Model. Lỗi: {e}")
            self.model = base_model

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        # Chuyển đổi bytes thành mảng numpy âm thanh bằng soundfile
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        
        # Đưa vào AI
        input_features = self.processor(audio_array, sampling_rate=sr, return_tensors="pt").input_features
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        
        return transcription

# Khởi tạo instance duy nhất để dùng chung cho mọi request
ai_service = DysarthriaModelService()
