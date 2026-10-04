from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

# ==========================================
# BÀI LAB 04: "PHÉP THUẬT" FINE-TUNING VỚI PEFT (LoRA)
# 
# Vấn đề: 
# Whisper đã rất giỏi nghe người bình thường nói. Nhưng bệnh nhân Dysarthria
# phát âm rất đặc thù. Nếu ta muốn dạy lại (Fine-tune) toàn bộ Whisper,
# thì máy tính bình thường sẽ "cháy" vì không đủ RAM/VRAM!
#
# Giải pháp: PEFT (Parameter-Efficient Fine-Tuning) - cụ thể là thuật toán LoRA.
# Thay vì đập đi xây lại, LoRA "đóng băng" 100% não bộ cũ của Whisper, 
# và chỉ cấy thêm một nhánh "tế bào nơ-ron mới" siêu nhỏ để học giọng Dysarthria.
# ==========================================

def print_trainable_parameters(model, label):
    """Hàm đếm số lượng nơ-ron (parameters) mà máy tính phải è cổ ra tính toán"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"\n[{label}]")
    print(f" - Tổng số tế bào (Total Parameters): {all_param:,}")
    print(f" - Số tế bào bị bắt đi học (Trainable): {trainable_params:,}")
    print(f" - Khối lượng công việc CPU/GPU phải gánh: {100 * trainable_params / all_param:.2f}%")

print("1. Tải Não bộ gốc (Whisper-tiny)...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

print_trainable_parameters(model, "TRƯỚC KHI DÙNG LORA (Mô hình gốc)")

print("\n2. Đang cấy ghép tế bào mới bằng LoRA...")
# Cấu hình LoRA: Chỉ cấy nơ-ron mới vào vùng "Chú ý" (Attention) của não bộ
config = LoraConfig(
    r=32, # Kích thước của mảng nơ-ron mới
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"], # Vùng cấy ghép
    lora_dropout=0.05,
    bias="none"
)

# Áp dụng phép thuật: Đóng băng não cũ + Lắp não mới vào
peft_model = get_peft_model(model, config)

print_trainable_parameters(peft_model, "SAU KHI ÁP DỤNG LORA")

print("\n=> KẾT LUẬN: Bằng cách này, khi bạn đưa lên Kaggle huấn luyện,")
print("=> GPU chỉ cần tối ưu hóa một phần siêu nhỏ (~3%) của mạng,")
print("=> Giúp tiết kiệm cực kỳ nhiều thời gian và tiền bạc mà độ chính xác vẫn ngang ngửa!")
