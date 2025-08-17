# generate_captions.py
from feature_extractor import FeatureExtractorBlip2
from PIL import Image
import json
import os
from tqdm import tqdm

# --- Cấu hình ---
TEST_IMG_DIR = "test2017/"  # Thư mục chứa ảnh COCO Test
TEST_ANNOTATIONS_FILE = "annotations/image_info_test2017.json"  # Annotations cho COCO Test
OUTPUT_CAPTIONS_FILE = "annotations/test2017_captions.json"  # File lưu captions
NUM_IMAGES = 500  # Số lượng ảnh để sinh captions

# Khởi tạo Blip2
try:
    fe = FeatureExtractorBlip2()
except Exception as e:
    print(f"LỖI: Không thể khởi tạo FeatureExtractorBlip2: {e}")
    exit(1)

# Tải danh sách ảnh Test
try:
    with open(TEST_ANNOTATIONS_FILE, "r") as f:
        test_data = json.load(f)
    test_images = test_data["images"][:NUM_IMAGES]
except Exception as e:
    print(f"LỖI: Không thể tải file annotations {TEST_ANNOTATIONS_FILE}: {e}")
    exit(1)

# Sinh captions
captions = []
for img_info in tqdm(test_images, desc="Đang sinh captions"):
    img_path = os.path.join(TEST_IMG_DIR, img_info["file_name"])
    try:
        image = Image.open(img_path).convert("RGB")
        caption = fe.generate_caption(image)
        captions.append({
            "image_id": img_info["id"],
            "file_name": img_info["file_name"],
            "caption": caption
        })
    except Exception as e:
        print(f"LỖI: Không thể sinh caption cho {img_path}: {e}")
        continue

# Lưu captions
try:
    with open(OUTPUT_CAPTIONS_FILE, "w") as f:
        json.dump(captions, f, indent=4)
    print(f"Đã lưu {len(captions)} captions vào {OUTPUT_CAPTIONS_FILE}")
except Exception as e:
    print(f"LỖI: Không thể lưu file {OUTPUT_CAPTIONS_FILE}: {e}")