"""
=============================================================
BẢN FULL CODE HUẤN LUYỆN (TRAINING SCRIPT) CHO KAGGLE / COLAB
=============================================================
Hướng dẫn: Bạn hãy copy toàn bộ file này, dán vào 1 ô (cell) trên Kaggle Notebook, 
bật GPU lên và bấm Run.
"""

import torch
import soundfile as sf
import io
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List, Union

print("1. TẢI VÀ GIẢI MÃ DỮ LIỆU...")
# Tải data. 
# Nếu bạn đã upload thẳng dataset này thành 1 mục bên phải Kaggle (như bạn nói),
# bạn không cần tải lại qua mạng nữa. Hãy copy đường dẫn thư mục /kaggle/input/... bỏ vào đây:
# Ví dụ:
dataset = load_dataset(
    "parquet", 
    data_files={
        "train": "/kaggle/input/datasets/alextranzz/dysarthria/train-00000-of-00001.parquet",
        "test": "/kaggle/input/datasets/alextranzz/dysarthria/test-00000-of-00001.parquet"
    }
)
dataset = dataset.cast_column("audio", Audio(decode=False))

print("2. TẢI NÃO BỘ WHISPER VÀ CẤY LORA...")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="en", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

config = LoraConfig(r=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)

print("3. CHUẨN BỊ DỮ LIỆU ĐỂ ĐƯA VÀO BĂNG CHUYỀN (PREPROCESSING)...")
def prepare_dataset(batch):
    # Giải mã thủ công để không bị treo
    audio_bytes = batch["audio"]["bytes"]
    audio_array, sr = sf.read(io.BytesIO(audio_bytes))
    
    # Chuyển thành ảnh Mel-Spectrogram (Ma trận)
    batch["input_features"] = processor.feature_extractor(audio_array, sampling_rate=sr).input_features[0]
    
    # Lấy text (Đáp án) -> Chuyển thành mã số Token
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

# Chạy lệnh chuyển đổi trên toàn bộ dataset (Đã bỏ num_proc=1 để chống lỗi treo đa luồng của Kaggle)
dataset = dataset.map(prepare_dataset)

print("4. CÀI ĐẶT BĂNG CHUYỀN DỮ LIỆU (DATA COLLATOR)...")
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

print("5. KHỞI ĐỘNG PHÒNG HUẤN LUYỆN (TRAINER)...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-dysarthria-lora",  # Nơi lưu trữ mô hình
    per_device_train_batch_size=8,           # Học 8 file cùng lúc
    learning_rate=1e-3,                      # Tốc độ học
    num_train_epochs=3,                      # Học đi học lại 3 vòng
    fp16=True,                               # Bật lõi tính toán siêu tốc của GPU
    logging_steps=5,                         # Cứ 5 bước in ra tiến độ 1 lần
    remove_unused_columns=False,             # Không xóa cột dữ liệu
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

print("\n=> 🚀 BẮT ĐẦU HUẤN LUYỆN (Training)...")
trainer.train()

print("\n=> 🎉 Đã học xong! Đang lưu bộ não mới...")
model.save_pretrained("LoRA_Dysarthria_Weights")
print("=> HOÀN TẤT! Hãy tìm thư mục 'LoRA_Dysarthria_Weights' nặng khoảng ~20MB bên trái và tải về máy tính!")
