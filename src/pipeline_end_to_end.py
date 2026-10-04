import os
from faster_whisper import WhisperModel
import librosa
import soundfile as sf
import ollama

# ==========================================
# FINAL LAB: PIPELINE THỰC TẾ TRỌN VẸN 
# (Âm thanh -> Nhận dạng siêu tốc -> Bác sĩ LLM chỉnh sửa)
# ==========================================

def run_dysarthria_pipeline(audio_path):
    print("\n--- BƯỚC 1: KHỞI ĐỘNG HỆ THỐNG ---")
    print("=> Tải bộ nhận diện âm thanh (Faster-Whisper)...")
    audio_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    
    print("\n--- BƯỚC 2: NGHE VÀ NHẬN DIỆN ÂM THANH ---")
    print(f"=> Đang phân tích file: {audio_path}")
    segments, info = audio_model.transcribe(audio_path, beam_size=5)
    
    # Gom các câu Whisper nghe được lại thành 1 đoạn văn bản
    raw_text = " ".join([segment.text for segment in segments]).strip()
    
    if not raw_text:
        print("[!] Whisper không nghe thấy tiếng người trong file này.")
        return
        
    print(f"\n[BẢN NHÁP CỦA WHISPER]: {raw_text}")
    
    print("\n--- BƯỚC 3: LLM BIÊN TẬP VÀ SỬA LỖI LÍU LƯỠI ---")
    print("=> Đang gửi bản nháp cho Qwen2.5 (Ollama)...")
    
    try:
        response = ollama.chat(
            model='qwen2.5:0.5b', 
            messages=[
                {
                    'role': 'system',
                    'content': 'Bạn là một bác sĩ ngôn ngữ. Nhiệm vụ của bạn là dịch và sửa lỗi chính tả/ngữ âm các câu nói bị líu lưỡi, nói ngọng của bệnh nhân thành một câu Tiếng Việt hoàn chỉnh, đúng chuẩn. CHỈ TRẢ LỜI CÂU ĐÃ SỬA, KHÔNG GIẢI THÍCH THÊM.'
                },
                {
                    'role': 'user',
                    'content': raw_text
                }
            ]
        )
        final_text = response['message']['content']
        print("\n==================================")
        print(f"✅ KẾT QUẢ CUỐI CÙNG DÀNH CHO BÁC SĨ ĐỌC:")
        print(f"👉 {final_text}")
        print("==================================")
        
    except Exception as e:
        print("\n[LỖI]: Không kết nối được với Ollama. Hãy chắc chắn ứng dụng Ollama đang chạy!")

if __name__ == "__main__":
    # GIẢ LẬP DỮ LIỆU: Tạo file âm thanh tạm thời
    # THỰC TẾ: Bạn hãy truyền đường dẫn file .wav ghi âm giọng của bạn vào hàm ở dưới
    print("Đang tạo file âm thanh mẫu (tiếng kèn trumpet)...")
    y, sr = librosa.load(librosa.ex('trumpet'), sr=16000)
    temp_wav = "test_voice.wav"
    sf.write(temp_wav, y, sr)
    
    # Chạy toàn bộ Pipeline
    run_dysarthria_pipeline(temp_wav)
    
    # Dọn dẹp
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
