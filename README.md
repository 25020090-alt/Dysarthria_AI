# Dysarthria AI - Speech Processing Project

Dự án Trí tuệ nhân tạo (AI) hỗ trợ xử lý và nhận dạng giọng nói dành cho người mắc chứng Loạn vận ngôn (Dysarthria). 

**Mục tiêu:** Sử dụng Deep Learning (Xử lý tín hiệu, PyTorch, Fine-tune các mô hình lớn như Whisper) để chuyển đổi và chẩn đoán giọng nói đặc thù của bệnh nhân Dysarthria.

---

## 🛠️ Hướng dẫn cài đặt (Installation Guide)

Để "người anh em" (bro) của bạn có thể lấy code về và chạy mượt mà, hãy làm đúng theo các bước sau:

### Bước 1: Lấy code về máy (Clone repo)
Mở Terminal ở thư mục bạn muốn lưu code và chạy:
```bash
git clone https://github.com/25020090-alt/Dysarthria_AI.git
cd Dysarthria_AI
```

### Bước 2: Tạo môi trường ảo (Virtual Environment)
Không cài trực tiếp vào Python gốc của máy để tránh xung đột thư viện.
```bash
python -m venv ai_env
```

### Bước 3: Kích hoạt môi trường ảo
- **Trên Windows (PowerShell/VS Code Terminal):**
  ```bash
  .\ai_env\Scripts\activate
  ```
- **Trên macOS / Linux:**
  ```bash
  source ai_env/bin/activate
  ```
*(Dấu hiệu thành công: Đầu dòng lệnh Terminal sẽ hiện chữ `(ai_env)`)*

### Bước 4: Cài đặt thư viện lõi
Đảm bảo bạn đang ở trong `ai_env`, sau đó chạy:
```bash
pip install -r requirements.txt
```

### Bước 5: Cài đặt PyTorch ("Trái tim" của AI)
Dự án sử dụng PyTorch. Chạy lệnh sau để cài đặt (phiên bản mặc định hỗ trợ CPU/CUDA):
```bash
pip install torch torchaudio
```

---

## 📁 Cấu trúc dự án
- `data/`: Nơi chứa dữ liệu âm thanh (`.wav`). Vì kích thước lớn nên đã bị chặn đẩy lên Git. Hãy tự tạo thư mục này ở local và bỏ file `.wav` test vào nhé.
- `models/`: Thư mục lưu trọng số model sau này.
- `notebooks/`: Chứa các file nháp (Jupyter Notebook).
- `src/`: 
  - `audio_matrix.py` (Lab 01): Code tiền xử lý âm thanh, bẻ 1 file `.wav` thành Ma trận ảnh phổ (Mel-spectrogram).
  - `dataset_loader.py` (Lab 02): Xây dựng kho chứa (`Dataset`) và băng chuyền (`DataLoader`) bằng PyTorch.

---

## 🚀 Chạy thử
Kiểm tra xem mọi thứ đã hoạt động chưa:
```bash
python src/audio_matrix.py
python src/dataset_loader.py
```
