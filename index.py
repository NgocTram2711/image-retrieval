# index.py
import os
import pickle
from tqdm import tqdm
import numpy as np
from feature_extractor import FeatureExtractorBlip2

# --- Cấu hình ---
DATASET_PATH = "dataset/"  # Đường dẫn đến thư mục chứa ảnh

# --- Bắt đầu quá trình ---
if __name__ == "__main__":
    fe = FeatureExtractorBlip2()

    # Danh sách để lưu đường dẫn ảnh và đặc trưng
    img_paths = []
    features_list = []

    print(f"Bắt đầu lập chỉ mục cho các ảnh trong thư mục '{DATASET_PATH}'...")
    
    # Kiểm tra xem thư mục dataset có tồn tại và có file ảnh không
    if not os.path.exists(DATASET_PATH):
        print(f"LỖI: Thư mục '{DATASET_PATH}' không tồn tại!")
        exit(1)
    
    # Lấy danh sách các file ảnh hợp lệ
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in os.listdir(DATASET_PATH) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not image_files:
        print(f"LỖI: Không tìm thấy ảnh hợp lệ trong thư mục '{DATASET_PATH}'!")
        exit(1)
    else:
        print(f"Tìm thấy {len(image_files)} ảnh hợp lệ.")

    # Duyệt qua từng ảnh và trích xuất đặc trưng
    for img_name in tqdm(image_files, desc="Đang trích xuất đặc trưng"):
        img_path = os.path.join(DATASET_PATH, img_name)
        try:
            feature = fe.extract_image_features(img_path)
            if feature is not None and feature.size > 0:
                features_list.append(feature)
                img_paths.append(img_path)
                # print(f"Đã xử lý ảnh: {img_path}, kích thước đặc trưng: {feature.shape}")
            else:
                print(f"CẢNH BÁO: Đặc trưng rỗng cho ảnh {img_path}")
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
    
    # Kiểm tra xem có đặc trưng nào được trích xuất không
    if not features_list:
        print(f"LỖI: Không trích xuất được đặc trưng nào!")
        exit(1)
    
    # Chuyển danh sách đặc trưng thành mảng numpy
    features_array = np.array(features_list)
    print(f"Kích thước mảng đặc trưng: {features_array.shape}")

    # Lưu kết quả
    with open(fe.FEATURES_FILE, "wb") as f:
        pickle.dump({
            "paths": img_paths,
            "features": features_array
        }, f)

    print(f"🎉 Hoàn tất! Đã lưu đặc trưng của {len(img_paths)} ảnh vào file '{fe.FEATURES_FILE}'.")