"""
=============================================================
BẢN CODE HUẤN LUYỆN BIG DATA (HÀNG CHỤC GB) TRÊN KAGGLE
=============================================================
Lưu ý: Dành riêng cho dữ liệu siêu lớn. Áp dụng cả 3 bí thuật:
1. Streaming (Không tải hết vào RAM)
2. Gradient Accumulation (Chống nổ VRAM)
3. Max Steps (Thay cho Epoch)
"""

import torch
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List, Union

print("1. KẾT NỐI STREAMING VỚI BIG DATA...")
# Bí thuật 1: streaming=True. Data sẽ chảy về từ từ như vòi nước.
dataset = load_dataset("Ankesh1234/dysarthria-asr-merged", streaming=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print("2. TẢI NÃO BỘ WHISPER VÀ CẤY LORA...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
config = LoraConfig(r=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)

print("3. CÀI ĐẶT BỘ CHUYỂN ĐỔI (Xử lý trực tiếp trên Stream)...")
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset)

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

print("4. MỞ PHÒNG HUẤN LUYỆN...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-bigdata-lora",
    per_device_train_batch_size=4,     # An toàn cho Kaggle T4 (Không bị nổ VRAM)
    gradient_accumulation_steps=4,     # Bí thuật 2: Cộng dồn 4 lần (4x4=16 file mỗi bước)
    learning_rate=1e-3,
    max_steps=5000,                    # Bí thuật 3: Bắt học 5000 bước rồi nghỉ (Vì streaming ko có điểm kết thúc rõ ràng)
    fp16=True,
    logging_steps=50,
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

print("\n=> 🚀 BẮT ĐẦU HUẤN LUYỆN TRÊN BIỂN DATA...")
trainer.train()

print("\n=> 🎉 Đã học xong! Đang lưu bộ não mới...")
model.save_pretrained("LoRA_BigData_Weights")
print("=> HOÀN TẤT! Tải thư mục 'LoRA_BigData_Weights' về máy tính nhé.")
