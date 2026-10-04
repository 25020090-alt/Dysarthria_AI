import ollama

# ==========================================
# BÀI LAB 06: "BÁC SĨ NGÔN NGỮ" (OLLAMA + LLM)
#
# Vấn đề: Dù Whisper có xịn đến đâu, tiếng người bệnh Dysarthria đôi khi 
# vẫn bị nuốt âm, khiến câu chữ sinh ra bị đứt đoạn hoặc sai chính tả.
# Ví dụ AI nghe được: "Tôi... mún... ún... nớc câm" 
#
# Giải pháp: Dùng một Mô hình Ngôn ngữ Lớn (LLM) như Qwen / Llama 
# đóng vai trò "Người biên tập". LLM hiểu ngữ cảnh siêu phàm, 
# nó sẽ tự động nối từ và sửa lại thành: "Tôi muốn uống nước cam."
# ==========================================

print("1. Chuẩn bị câu văn bị lỗi...")
# Giả lập kết quả đầu ra bị lỗi từ Faster-Whisper
broken_sentence = "tôi mún ún nớc câm"

print(f"=> Câu gốc (do phát âm méo): '{broken_sentence}'")
print("\n2. Đang gửi cho Bác sĩ LLM (Qwen) để khám và sửa lỗi...")

try:
    # Gọi Ollama chạy ngầm dưới máy tính của bạn
    # Dùng qwen2.5:0.5b (mô hình siêu nhẹ, siêu nhanh)
    response = ollama.chat(
        model='qwen2.5:0.5b', 
        messages=[
            {
                'role': 'system',
                'content': 'Bạn là một bác sĩ ngôn ngữ. Nhiệm vụ của bạn là dịch và sửa lỗi chính tả/ngữ âm các câu nói bị líu lưỡi, nói ngọng của bệnh nhân thành một câu Tiếng Việt hoàn chỉnh, đúng chuẩn. CHỈ TRẢ LỜI CÂU ĐÃ SỬA, KHÔNG GIẢI THÍCH THÊM.'
            },
            {
                'role': 'user',
                'content': broken_sentence
            }
        ]
    )
    
    print("\n==================================")
    print(f"✨ CÂU ĐÃ ĐƯỢC LLM SỬA LẠI: {response['message']['content']}")
    print("==================================")
    
except Exception as e:
    print("\n[LỖI]: Không tìm thấy ứng dụng Ollama đang chạy.")
    print("-> BƯỚC CẦN LÀM: ")
    print("1. Tải và cài đặt phần mềm Ollama tại: https://ollama.com")
    print("2. Mở một Terminal hệ thống, gõ lệnh: ollama run qwen2.5:0.5b")
    print("3. Sau khi nó tải xong 300MB, quay lại đây chạy script này là được!")
