# evaluate.py
import pickle
import numpy as np
import os
from pycocotools.coco import COCO
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from feature_extractor import FeatureExtractorBlip2

# --- Cấu hình ---
ANNOTATIONS_FILE = "annotations/captions_val2017.json"  # Tải từ https://cocodataset.org/#download
# ANNOTATIONS_FILE = "annotations/captions_train2017.json"  # Tải từ https://cocodataset.org/#download
NUM_QUERIES = 50  # Tối thiểu 50 queries theo yêu cầu bài tập
DATASET_PATH = "dataset/"  # Thư mục chứa ảnh MS COCO (val2017)

def extract_image_id_from_path(path):
    """Trích xuất image_id từ filename, ví dụ: '000000123456.jpg' -> 123456"""
    filename = os.path.basename(path)
    return int(filename.split('.')[0].lstrip('0'))

if __name__ == "__main__":
    fe = FeatureExtractorBlip2()

    # Tải features
    with open(fe.FEATURES_FILE, "rb") as f:
        data = pickle.load(f)
    img_paths = data["paths"]
    features_array = data["features"]

    # Map img_paths đến image_ids
    path_to_id = {path: extract_image_id_from_path(path) for path in img_paths}

    # Tải COCO annotations
    coco = COCO(ANNOTATIONS_FILE)

    # Chọn 50 queries ngẫu nhiên (mỗi query là một caption và ground truth image_id)
    # Lấy tất cả annotations, chọn unique images, rồi chọn 50
    img_ids = list(coco.imgs.keys())[:NUM_QUERIES]  # Lấy 50 images đầu tiên cho đơn giản
    queries = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if ann_ids:
            caption = coco.anns[ann_ids[0]]['caption']  # Lấy caption đầu tiên
            queries.append((caption, img_id))

    print(f"Đánh giá trên {len(queries)} queries từ MS COCO val2017...")

    # Tính mAP
    aps = []
    for text, gt_id in tqdm(queries, desc="Đang đánh giá"):
        # Trích xuất features cho text query
        q_feat = fe.extract_text_features(text)
        
        # Tính similarities
        sims = np.dot(features_array, q_feat.T).squeeze()
        
        # Tạo y_true
        y_true = np.array([1 if path_to_id[path] == gt_id else 0 for path in img_paths])
        
        # --- THÊM ĐOẠN CODE DEBUG NÀY VÀO ---
        if np.sum(y_true) == 0:
            print(f"!!! CẢNH BÁO: Không tìm thấy ảnh đáp án với ID {gt_id} trong bộ dữ liệu đã index.")
            aps.append(0.0)  # Gán AP bằng 0 cho query này
            continue         # Bỏ qua và tiếp tục với query tiếp theo
        # --- KẾT THÚC ĐOẠN CODE DEBUG ---

        # Tính AP
        ap = average_precision_score(y_true, sims)
        aps.append(ap)

    mean_ap = np.mean(aps)
    print(f"Mean Average Precision (mAP) cho text-to-image retrieval: {mean_ap:.4f}")

    # Optional: Thêm Recall@K
    recall_at_10 = []
    for text, gt_id in tqdm(queries, desc="Đang tính Recall@10"):
        q_feat = fe.extract_text_features(text)
        sims = np.dot(features_array, q_feat.T).squeeze() if features_array.ndim == 2 else np.dot(features_array, q_feat)
        top_indices = np.argsort(-sims)[:10]
        top_ids = [path_to_id[img_paths[idx]] for idx in top_indices]
        recall_at_10.append(1 if gt_id in top_ids else 0)

    mean_recall_10 = np.mean(recall_at_10)
    print(f"Mean Recall@10: {mean_recall_10:.4f}")