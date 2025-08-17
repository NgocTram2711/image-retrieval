import pickle
import numpy as np
import os
import json
import torch
from pycocotools.coco import COCO
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from feature_extractor import FeatureExtractorClip, FeatureExtractorSiglip2, FeatureExtractorBlip2
import argparse
import ollama  # Thêm Ollama để tinh chỉnh caption

# --- Cấu hình ---
VAL_ANNOTATIONS_FILE = "annotations/captions_val2017.json"  # Annotations cho COCO Val captions
TEST_ANNOTATIONS_FILE = "annotations/image_info_test2017.json"  # Annotations cho COCO Test (image queries)
TEST_CAPTIONS_FILE = "annotations/test2017_captions.json"  # Captions sinh ra cho COCO Test
TEST_IMG_DIR = "test2017/"  # Thư mục chứa ảnh COCO Test
NUM_QUERIES = 500  # Số lượng queries
DATASET_PATH = "dataset/"  # Thư mục chứa ảnh COCO Val

def extract_image_id_from_path(path):
    """Trích xuất image_id từ filename, ví dụ: '000000123456.jpg' -> 123456"""
    filename = os.path.basename(path)
    return int(filename.split('.')[0].lstrip('0'))

def refine_caption(caption, model="llama3", max_length=60):
    """Tinh chỉnh caption bằng Ollama, đảm bảo độ dài dưới max_length ký tự."""
    prompt = (
        f"Refine this image caption to make it more descriptive but concise (max {max_length} characters) "
        f"for better image search: '{caption}'"
    )
    try:
        response = ollama.generate(model=model, prompt=prompt)
        refined = response['response'].strip()
        # Cắt ngắn nếu vẫn vượt quá
        if len(refined) > max_length:
            refined = refined[:max_length].rsplit(' ', 1)[0] + '...'
        return refined
    except Exception as e:
        print(f"LỖI: Không thể tinh chỉnh caption '{caption}': {e}")
        return caption  # Giữ nguyên nếu lỗi

def main(extractor_name):
    # Khởi tạo feature extractor
    if extractor_name == "clip":
        fe = FeatureExtractorClip()
        features_file = "features_clip.pkl"
        result_file = "result_clip.json"
        expected_feature_size = 768
        max_token_length = 77  # Giới hạn token của CLIP
    elif extractor_name == "siglip":
        fe = FeatureExtractorSiglip2()
        features_file = "features_siglip2.pkl"
        result_file = "result_siglip.json"
        expected_feature_size = 1024
        max_token_length = 64  # Giới hạn token của SigLIP
    elif extractor_name == "blip":
        fe = FeatureExtractorBlip2()
        features_file = "features_blip2.pkl"
        result_file = "result_blip.json"
        expected_feature_size = 768
        max_token_length = 77  # Giới hạn token của Blip2 (tương tự CLIP)
    else:
        print(f"LỖI: Feature extractor '{extractor_name}' không được hỗ trợ. Chọn 'clip', 'siglip', hoặc 'blip'.")
        exit(1)

    # Tải features của COCO Val
    try:
        with open(features_file, "rb") as f:
            data = pickle.load(f)
        img_paths = data["paths"]
        features_array = data["features"]
        print(f"Loaded {len(img_paths)} image features with shape {features_array.shape} from {features_file}")
    except Exception as e:
        print(f"LỖI: Không thể tải file {features_file}: {e}")
        exit(1)

    # Kiểm tra kích thước đặc trưng ảnh
    if features_array.shape[1] != expected_feature_size:
        print(f"LỖI: Kích thước đặc trưng ảnh ({features_array.shape[1]}) không khớp với kích thước mong đợi ({expected_feature_size}). Vui lòng chạy lại index.py.")
        exit(1)

    # Map img_paths đến image_ids (COCO Val)
    path_to_id = {path: extract_image_id_from_path(path) for path in img_paths}

    # Tải COCO Val annotations
    try:
        coco_val = COCO(VAL_ANNOTATIONS_FILE)
    except Exception as e:
        print(f"LỖI: Không thể tải file annotations {VAL_ANNOTATIONS_FILE}: {e}")
        exit(1)

    # Tải COCO Test annotations
    try:
        coco_test = COCO(TEST_ANNOTATIONS_FILE)
    except Exception as e:
        print(f"LỖI: Không thể tải file annotations {TEST_ANNOTATIONS_FILE}: {e}")
        exit(1)

    # Tải captions sinh ra cho COCO Test
    try:
        with open(TEST_CAPTIONS_FILE, "r") as f:
            test_captions = json.load(f)
    except Exception as e:
        print(f"LỖI: Không thể tải file captions {TEST_CAPTIONS_FILE}: {e}")
        exit(1)

    # Chọn NUM_QUERIES ảnh từ COCO Test cho image-to-image
    test_img_ids = list(coco_test.imgs.keys())[:NUM_QUERIES]
    image_queries = []
    for img_id in test_img_ids:
        img_info = coco_test.imgs[img_id]
        img_path = os.path.join(TEST_IMG_DIR, img_info["file_name"])
        if os.path.exists(img_path):
            image_queries.append((img_path, img_id))
        else:
            print(f"CẢNH BÁO: Không tìm thấy ảnh {img_path}")

    # Chọn NUM_QUERIES captions từ test_captions cho text-to-image và tinh chỉnh
    text_queries = [(cap["caption"], cap["image_id"]) for cap in test_captions[:NUM_QUERIES]]
    
    # Tính trước đặc trưng cho tất cả captions Val
    val_captions = []
    val_caption_to_img_id = []
    for img_id in coco_val.imgs.keys():
        ann_ids = coco_val.getAnnIds(imgIds=img_id)
        if ann_ids:
            # Chọn caption đầu tiên mỗi ảnh để giảm tải
            caption = coco_val.anns[ann_ids[0]]['caption']
            val_captions.append(caption)
            val_caption_to_img_id.append(img_id)

    print(f"Tính trước đặc trưng cho {len(val_captions)} captions Val...")
    val_caption_features = []
    batch_size = 32
    for i in tqdm(range(0, len(val_captions), batch_size), desc="Trích xuất đặc trưng Val captions"):
        batch_captions = val_captions[i:i + batch_size]
        # Cắt ngắn caption nếu cần
        batch_captions = [cap[:max_token_length] for cap in batch_captions]
        try:
            inputs = fe.processor(text=batch_captions, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length).to(fe.device)
            with torch.no_grad():
                if extractor_name == "blip":
                    # Blip2 sử dụng Q-Former để trích xuất đặc trưng văn bản
                    inputs["pixel_values"] = fe.dummy_image.repeat(len(batch_captions), 1, 1, 1)
                    outputs = fe.model(**inputs, return_dict=True)
                    features = outputs.qformer_outputs.pooler_output
                else:
                    # Clip và Siglip sử dụng get_text_features
                    features = fe.model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            val_caption_features.extend(features.cpu().numpy())
        except Exception as e:
            print(f"LỖI: Không thể trích xuất đặc trưng cho batch captions: {e}")
            val_caption_features.extend([np.zeros(expected_feature_size)] * len(batch_captions))

    val_caption_features = np.array(val_caption_features)
    print(f"Đã tính đặc trưng cho {len(val_caption_features)} captions Val với shape {val_caption_features.shape}")

    # Kiểm tra kích thước đặc trưng văn bản
    if val_caption_features.shape[1] != expected_feature_size:
        print(f"LỖI: Kích thước đặc trưng văn bản ({val_caption_features.shape[1]}) không khớp với kích thước đặc trưng ảnh ({expected_feature_size}).")
        exit(1)

    print(f"Đánh giá trên {len(text_queries)} text queries và {len(image_queries)} image queries từ COCO 2017 Test...")

    # --- Text-to-Image Retrieval Evaluation ---
    aps_text = []
    for text, gt_id in tqdm(text_queries, desc="Đang đánh giá Text-to-Image"):
        try:
            # Cắt ngắn caption nếu cần
            text = text[:max_token_length]
            q_feat = fe.extract_text_features(text)
            if q_feat.shape[0] != expected_feature_size:
                print(f"CẢNH BÁO: Kích thước đặc trưng văn bản không khớp: {q_feat.shape[0]} (mong đợi: {expected_feature_size})")
                aps_text.append(0.0)
                continue

            sims = np.dot(features_array, q_feat.T).squeeze() if features_array.ndim == 2 else np.dot(features_array, q_feat)

            # Ground truth: Tìm caption Val giống nhất
            caption_sims = np.dot(val_caption_features, q_feat)
            relevant_idx = np.argmax(caption_sims)
            relevant_img_id = val_caption_to_img_id[relevant_idx]
            y_true = np.array([1 if path_to_id[path] == relevant_img_id else 0 for path in img_paths])

            ap = average_precision_score(y_true, sims)
            aps_text.append(ap)

        except Exception as e:
            print(f"LỖI: Không thể trích xuất đặc trưng văn bản cho '{text}': {e}")
            aps_text.append(0.0)
            continue

    mean_ap_text = np.mean(aps_text)
    print(f"Mean Average Precision (mAP) cho text-to-image retrieval: {mean_ap_text:.4f}")

    recall_at_10_text = []
    for text, gt_id in tqdm(text_queries, desc="Đang tính Recall@10 Text-to-Image"):
        try:
            # Cắt ngắn caption nếu cần
            text = text[:max_token_length]
            q_feat = fe.extract_text_features(text)
            if q_feat.shape[0] != expected_feature_size:
                print(f"CẢNH BÁO: Kích thước đặc trưng văn bản không khớp: {q_feat.shape[0]} (mong đợi: {expected_feature_size})")
                recall_at_10_text.append(0.0)
                continue

            sims = np.dot(features_array, q_feat.T).squeeze() if features_array.ndim == 2 else np.dot(features_array, q_feat)
            top_indices = np.argsort(-sims)[:10]
            caption_sims = np.dot(val_caption_features, q_feat)
            relevant_idx = np.argmax(caption_sims)
            relevant_img_id = val_caption_to_img_id[relevant_idx]
            top_img_ids = [path_to_id[img_paths[idx]] for idx in top_indices]
            recall_at_10_text.append(1 if relevant_img_id in top_img_ids else 0)

        except Exception as e:
            print(f"LỖI: Không thể trích xuất đặc trưng văn bản cho '{text}': {e}")
            recall_at_10_text.append(0.0)
            continue

    mean_recall_10_text = np.mean(recall_at_10_text)
    print(f"Mean Recall@10 cho text-to-image retrieval: {mean_recall_10_text:.4f}")

    # --- Image-to-Image Retrieval Evaluation ---
    aps_image = []
    for img_path, gt_id in tqdm(image_queries, desc="Đang đánh giá Image-to-Image"):
        try:
            q_feat = fe.extract_image_features(img_path)
            if q_feat.shape[0] != expected_feature_size:
                print(f"CẢNH BÁO: Kích thước đặc trưng ảnh không khớp: {q_feat.shape[0]} (mong đợi: {expected_feature_size})")
                aps_image.append(0.0)
                continue

            sims = np.dot(features_array, q_feat.T).squeeze() if features_array.ndim == 2 else np.dot(features_array, q_feat)

            # Ground truth: Chọn ảnh Val có similarity cao nhất
            top_idx = np.argmax(sims)
            y_true = np.zeros(len(img_paths))
            y_true[top_idx] = 1

            ap = average_precision_score(y_true, sims)
            aps_image.append(ap)

        except Exception as e:
            print(f"LỖI: Không thể trích xuất đặc trưng cho ảnh {img_path}: {e}")
            aps_image.append(0.0)
            continue

    mean_ap_image = np.mean(aps_image)
    print(f"Mean Average Precision (mAP) cho image-to-image retrieval: {mean_ap_image:.4f}")

    recall_at_10_image = []
    for img_path, gt_id in tqdm(image_queries, desc="Đang tính Recall@10 Image-to-Image"):
        try:
            q_feat = fe.extract_image_features(img_path)
            if q_feat.shape[0] != expected_feature_size:
                print(f"CẢNH BÁO: Kích thước đặc trưng ảnh không khớp: {q_feat.shape[0]} (mong đợi: {expected_feature_size})")
                recall_at_10_image.append(0.0)
                continue

            sims = np.dot(features_array, q_feat.T).squeeze() if features_array.ndim == 2 else np.dot(features_array, q_feat)
            top_indices = np.argsort(-sims)[:10]
            top_idx = np.argmax(sims)
            recall_at_10_image.append(1 if top_idx in top_indices else 0)

        except Exception as e:
            print(f"LỖI: Không thể trích xuất đặc trưng cho ảnh {img_path}: {e}")
            recall_at_10_image.append(0.0)
            continue

    mean_recall_10_image = np.mean(recall_at_10_image)
    print(f"Mean Recall@10 cho image-to-image retrieval: {mean_recall_10_image:.4f}")

    # Lưu kết quả
    results = {
        "extractor": extractor_name,
        "text_to_image": {
            "mean_ap": float(mean_ap_text),
            "mean_recall_at_10": float(mean_recall_10_text)
        },
        "image_to_image": {
            "mean_ap": float(mean_ap_image),
            "mean_recall_at_10": float(mean_recall_10_image)
        }
    }
    try:
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Đã lưu kết quả vào {result_file}")
    except Exception as e:
        print(f"LỖI: Không thể lưu file kết quả {result_file}: {e}")

    # Xóa bộ nhớ GPU
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá image retrieval với feature extractor.")
    parser.add_argument("--extractor", type=str, default="clip", choices=["clip", "siglip", "blip"], help="Feature extractor: clip, siglip, hoặc blip.")
    args = parser.parse_args()
    main(args.extractor)