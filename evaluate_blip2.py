import pickle
import numpy as np
import os
import json
import torch
from pycocotools.coco import COCO
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from feature_extractor import FeatureExtractorBlip2

# --- Cấu hình ---
VAL_ANNOTATIONS_FILE = "annotations/captions_val2017.json"
TEST_CAPTIONS_FILE = "annotations/test2017_captions.json"
DB_CAPTIONS_FILE = "db_captions.json" # Dành cho Text-to-Image
FEATURES_FILE = "features_blip2.pkl" # Dành cho Image-to-Image
NUM_QUERIES = 500
TEST_IMG_DIR = "test2017/"

def extract_image_id_from_path(path):
    """Trích xuất image_id từ filename."""
    filename = os.path.basename(path)
    return int(filename.split('.')[0])

def main():
    print("🚀 Bắt đầu quá trình đánh giá chỉ dành cho BLIP-2...")
    
    # --- 1. Khởi tạo các model cần thiết ---
    fe = FeatureExtractorBlip2() # Model BLIP-2 để trích xuất đặc trưng hình ảnh
    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Đã tải các model cần thiết.")

    # --- 2. Tải và chuẩn bị Database ---
    
    # 2.1. Tải Database cho Image-to-Image
    with open(FEATURES_FILE, "rb") as f:
        data = pickle.load(f)
    db_img_paths = data["paths"]
    db_image_features = data["features"]
    db_path_to_id = {path: extract_image_id_from_path(path) for path in db_img_paths}
    print(f"✅ Đã tải {len(db_img_paths)} features ẢNH cho tác vụ Image-to-Image.")

    # 2.2. Tải và chuẩn bị Database cho Text-to-Image
    with open(DB_CAPTIONS_FILE, "r", encoding='utf-8') as f:
        db_captions_data = json.load(f)
    path_to_caption = {item['image_path']: item['caption'] for item in db_captions_data}
    ordered_db_captions = [path_to_caption.get(p, "") for p in db_img_paths]
    db_caption_embeddings = st_model.encode(ordered_db_captions, convert_to_tensor=True, show_progress_bar=True, desc="Mã hóa DB Captions")
    print(f"✅ Đã tải và mã hóa {len(ordered_db_captions)} CAPTIONS cho tác vụ Text-to-Image.")

    # --- 3. Tải dữ liệu truy vấn (Queries) và Ground Truth ---
    coco_val = COCO(VAL_ANNOTATIONS_FILE)
    with open(TEST_CAPTIONS_FILE, "r") as f:
        test_captions = json.load(f)
    
    text_queries = [(cap["caption"], cap["image_id"]) for cap in test_captions[:NUM_QUERIES]]
    image_queries = [(os.path.join(TEST_IMG_DIR, item["file_name"]), item["image_id"]) for item in test_captions[:NUM_QUERIES]]

    # Tính trước embedding cho Val Captions để xác định ground truth
    val_captions = [ann['caption'] for ann in coco_val.anns.values()]
    val_caption_img_ids = [ann['image_id'] for ann in coco_val.anns.values()]
    val_caption_embeddings = st_model.encode(val_captions, convert_to_tensor=True, show_progress_bar=True, desc="Mã hóa Val Captions (GT)")

    # --- 4. Đánh giá Text-to-Image (SỬ DỤNG SENTENCE TRANSFORMER VÀ CAPTIONS) ---
    print("\nBắt đầu đánh giá Text-to-Image (phương pháp Text-vs-Text)...")
    aps_text, recall_at_10_text = [], []

    for text, _ in tqdm(text_queries, desc="Đánh giá Text-to-Image"):
        try:
            query_embedding_st = st_model.encode(text, convert_to_tensor=True)
            gt_sims = util.cos_sim(query_embedding_st, val_caption_embeddings)[0]
            relevant_idx = torch.argmax(gt_sims).item()
            relevant_img_id = val_caption_img_ids[relevant_idx]
            y_true = np.array([1 if db_path_to_id.get(path) == relevant_img_id else 0 for path in db_img_paths])

            if not np.any(y_true): continue

            search_hits = util.semantic_search(query_embedding_st, db_caption_embeddings, top_k=len(db_img_paths))[0]
            sims = np.zeros(len(db_img_paths)); top_indices = []
            for hit in search_hits: sims[hit['corpus_id']] = hit['score']
            top_indices = [hit['corpus_id'] for hit in search_hits[:10]]

            ap = average_precision_score(y_true, sims)
            aps_text.append(ap)
            top_img_ids = [db_path_to_id[db_img_paths[idx]] for idx in top_indices]
            recall_at_10_text.append(1 if relevant_img_id in top_img_ids else 0)
        except Exception as e:
            print(f"\nLỗi khi xử lý query text '{text}': {e}"); aps_text.append(0.0); recall_at_10_text.append(0.0)

    mean_ap_text = np.mean(aps_text) if aps_text else 0.0
    mean_recall_10_text = np.mean(recall_at_10_text) if recall_at_10_text else 0.0

    # --- 5. Đánh giá Image-to-Image (CHỈ SỬ DỤNG VECTOR ẢNH TỪ BLIP-2) ---
    print(f"\nBắt đầu đánh giá Image-to-Image...")
    aps_image, recall_at_10_image = [], []
    for img_path, gt_id in tqdm(image_queries, desc="Đánh giá Image-to-Image"):
        try:
            q_feat = fe.extract_image_features(img_path)
            sims = np.dot(db_image_features, q_feat)
            relevant_idx = np.argmax(sims)
            relevant_img_id = db_path_to_id[db_img_paths[relevant_idx]]
            y_true = np.array([1 if db_path_to_id.get(path) == relevant_img_id else 0 for path in db_img_paths])

            ap = average_precision_score(y_true, sims)
            aps_image.append(ap)
            top_indices = np.argsort(-sims)[:10]
            top_img_ids = [db_path_to_id[db_img_paths[idx]] for idx in top_indices]
            recall_at_10_image.append(1 if relevant_img_id in top_img_ids else 0)
        except Exception as e:
            print(f"\nLỗi khi xử lý query image '{img_path}': {e}"); aps_image.append(0.0); recall_at_10_image.append(0.0)

    mean_ap_image = np.mean(aps_image) if aps_image else 0.0
    mean_recall_10_image = np.mean(recall_at_10_image) if recall_at_10_image else 0.0

    # --- 6. Lưu kết quả ---
    result_file = "result_blip2.json"
    results = {"extractor": "blip2_caption_search", "text_to_image": {"mean_ap": float(mean_ap_text), "mean_recall_at_10": float(mean_recall_10_text)}, "image_to_image": {"mean_ap": float(mean_ap_image), "mean_recall_at_10": float(mean_recall_10_image)}}
    with open(result_file, "w") as f: json.dump(results, f, indent=4)
    print("\n--- KẾT QUẢ CUỐI CÙNG ---"); print(json.dumps(results, indent=4)); print(f"Đã lưu kết quả vào {result_file}")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()