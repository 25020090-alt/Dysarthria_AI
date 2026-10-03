import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa

# ==========================================
# KHÁI NIỆM 1: DATASET (KHO CHỨA & BIẾN ĐỔI)
# ==========================================
class DysarthriaDataset(Dataset):
    def __init__(self, file_paths):
        """
        Khởi tạo kho chứa. 
        Truyền vào danh sách đường dẫn tới các file .wav
        """
        self.file_paths = file_paths

    def __len__(self):
        """Khai báo cho PyTorch biết kho này có bao nhiêu file"""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        TRÁI TIM CỦA BĂNG CHUYỀN:
        Mỗi khi PyTorch hô "Lấy cho tao file số idx!", hàm này sẽ chạy ngầm.
        Nó sẽ lấy file đó, chế biến thành Ma trận, rồi ném vào cho AI.
        """
        file_path = self.file_paths[idx]
        
        # 1. Đọc âm thanh thô
        y, sr = librosa.load(file_path, sr=22050)
        
        # 2. Biến thành Ma trận ảnh (giống Bài Lab 01)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 3. Ép kiểu: Từ Ma trận Numpy (Python thường) sang Tensor (ngôn ngữ của PyTorch)
        #unsqueeze(0) dùng để thêm 1 chiều nữa (gọi là 'channel' - giống ảnh trắng đen có 1 kênh màu)
        tensor_data = torch.FloatTensor(S_dB).unsqueeze(0) 
        
        return tensor_data

# ==========================================
# KHÁI NIỆM 2: DATALOADER (BĂNG CHUYỀN LẮP RÁP)
# ==========================================
if __name__ == "__main__":
    print("Đang giả lập danh sách file...")
    
    # Ở đây thay vì dùng file thật, mình tạm mượn file trumpet 3 lần 
    # coi như giả lập ta có 3 file ghi âm của 3 bệnh nhân khác nhau.
    fake_files = [librosa.ex('trumpet'), librosa.ex('trumpet'), librosa.ex('trumpet')]
    
    # 1. Tạo kho chứa (Dataset)
    dataset = DysarthriaDataset(fake_files)
    print(f"Tổng số file ghi âm trong kho: {len(dataset)}")
    
    # 2. Tạo băng chuyền (DataLoader)
    # Nhiệm vụ: Tự động nhặt ngẫu nhiên dữ liệu từ Dataset, 
    # gom thành từng Lô (Batch) để AI học cho nhanh, thay vì học từng file một.
    # batch_size=2 nghĩa là mỗi lô đóng gói 2 file.
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Giả lập quá trình AI đang học
    print("\n--- Bắt đầu cho AI ăn dữ liệu ---")
    for batch_idx, batch_data in enumerate(dataloader):
        print(f"\n=> Lô (Batch) thứ {batch_idx + 1}:")
        print(f"   Kích thước Tensor của lô này: {batch_data.shape}")
        
        # Nhìn vào kích thước Tensor bạn sẽ thấy:
        # [2, 1, 128, 230] = [Số file trong lô, Số kênh màu, Số dải tần số, Độ dài thời gian]
        
        break # Chỉ chạy 1 lô để xem thử cấu trúc thôi
