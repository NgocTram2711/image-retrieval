# index.py
import os
import pickle
from tqdm import tqdm
import numpy as np
from feature_extractor import FeatureExtractorClip

# --- Cấu hình ---
DATASET_PATH = "dataset/"  # Đường dẫn đến thư mục chứa ảnh

# --- Bắt đầu quá trình ---
if __name__ == "__main__":
    fe = FeatureExtractorClip()

    # Danh sách để lưu đường dẫn ảnh và đặc trưng
    img_paths = []
    features_list = []

    print(f"Bắt đầu lập chỉ mục cho các ảnh trong thư mục '{DATASET_PATH}'...")
    
    # Lấy danh sách các file ảnh hợp lệ
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in os.listdir(DATASET_PATH) if os.path.splitext(f)[1].lower() in valid_extensions]

    # Duyệt qua từng ảnh và trích xuất đặc trưng
    for img_name in tqdm(image_files, desc="Đang trích xuất đặc trưng"):
        img_path = os.path.join(DATASET_PATH, img_name)
        try:
            feature = fe.extract_image_features(img_path)
            features_list.append(feature)
            img_paths.append(img_path)
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
    
    # Chuyển danh sách đặc trưng thành mảng numpy
    features_array = np.array(features_list)

    # Lưu kết quả
    with open(fe.FEATURES_FILE, "wb") as f:
        pickle.dump({
            "paths": img_paths,
            "features": features_array
        }, f)

    print(f"🎉 Hoàn tất! Đã lưu đặc trưng của {len(img_paths)} ảnh vào file '{fe.FEATURES_FILE}'.")