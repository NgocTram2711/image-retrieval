import os
import json
from PIL import Image
from tqdm import tqdm
from feature_extractor import FeatureExtractorBlip2

# --- Cấu hình ---
DATASET_PATH = "dataset/"  # Đường dẫn đến thư mục chứa ảnh
OUTPUT_FILE = "db_captions.json" # File để lưu trữ caption

print("Bắt đầu quá trình tạo caption cho database...")

# 1. Khởi tạo model BLIP-2
try:
    fe = FeatureExtractorBlip2()
    print("✅ Model BLIP-2 đã được tải thành công.")
except Exception as e:
    print(f"LỖI: Không thể khởi tạo FeatureExtractorBlip2: {e}")
    exit(1)

# 2. Lấy danh sách tất cả các file ảnh hợp lệ
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
try:
    image_files = [f for f in os.listdir(DATASET_PATH) if os.path.splitext(f)[1].lower() in valid_extensions]
    if not image_files:
        print(f"LỖI: Không tìm thấy ảnh nào trong '{DATASET_PATH}'!")
        exit(1)
    print(f"Tìm thấy {len(image_files)} ảnh để xử lý.")
except FileNotFoundError:
    print(f"LỖI: Thư mục '{DATASET_PATH}' không tồn tại!")
    exit(1)

# 3. Duyệt qua từng ảnh và sinh caption
database_captions = []
for img_name in tqdm(image_files, desc="Đang sinh caption"):
    img_path = os.path.join(DATASET_PATH, img_name)
    try:
        image = Image.open(img_path).convert("RGB")
        # Sử dụng hàm generate_caption có sẵn trong FeatureExtractorBlip2
        caption = fe.generate_caption(image)
        
        database_captions.append({
            "image_path": img_path,
            "caption": caption
        })
    except Exception as e:
        print(f"\nLỗi khi xử lý ảnh {img_path}: {e}")
        continue

# 4. Lưu kết quả ra file JSON
try:
    with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
        json.dump(database_captions, f, indent=4, ensure_ascii=False)
    print(f"\n🎉 Hoàn tất! Đã lưu {len(database_captions)} captions vào file '{OUTPUT_FILE}'.")
except Exception as e:
    print(f"\nLỖI: Không thể lưu file kết quả '{OUTPUT_FILE}': {e}")