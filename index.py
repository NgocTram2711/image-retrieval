# index.py
import os
import pickle
from tqdm import tqdm
import numpy as np
import argparse
from feature_extractor import FeatureExtractorClip, FeatureExtractorBlip2, FeatureExtractorSiglip2

# --- Cấu hình ---
DATASET_PATH = "dataset/"  # Đường dẫn đến thư mục chứa ảnh

# --- Bắt đầu quá trình ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lập chỉ mục hình ảnh bằng một feature extractor cụ thể.")
    parser.add_argument("--model", type=str, required=True, choices=["clip", "blip", "siglip"], 
                        help="Model để sử dụng: 'clip', 'blip', hoặc 'siglip'.")
    args = parser.parse_args()

    # Chọn feature extractor dựa trên tham số
    if args.model == "clip":
        fe = FeatureExtractorClip()
    elif args.model == "blip":
        fe = FeatureExtractorBlip2()
    elif args.model == "siglip":
        fe = FeatureExtractorSiglip2()
    else:
        # Dòng này thực tế sẽ không chạy được do 'choices' trong argparse
        print(f"LỖI: Model '{args.model}' không được hỗ trợ.")
        exit(1)

    # Danh sách để lưu đường dẫn ảnh và đặc trưng
    img_paths = []
    features_list = []

    print(f"Bắt đầu lập chỉ mục cho các ảnh trong thư mục '{DATASET_PATH}' sử dụng model {args.model.upper()}...")
    
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
    for img_name in tqdm(image_files, desc=f"Đang trích xuất đặc trưng ({args.model.upper()})"):
        img_path = os.path.join(DATASET_PATH, img_name)
        try:
            # Tất cả các class extractor đều có cùng phương thức `extract_image_features`
            feature = fe.extract_image_features(img_path)
            if feature is not None and feature.size > 0:
                features_list.append(feature)
                img_paths.append(img_path)
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

    # Lưu kết quả vào file tương ứng với model (fe.FEATURES_FILE)
    with open(fe.FEATURES_FILE, "wb") as f:
        pickle.dump({
            "paths": img_paths,
            "features": features_array
        }, f)

    print(f"🎉 Hoàn tất! Đã lưu đặc trưng của {len(img_paths)} ảnh vào file '{fe.FEATURES_FILE}'.")