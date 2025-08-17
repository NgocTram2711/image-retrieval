import json
from sentence_transformers import SentenceTransformer, util
import torch

# --- Cấu hình ---
CAPTIONS_DB_FILE = "db_captions.json"
# Model chuyên dụng để so sánh sự tương đồng ngữ nghĩa giữa các câu
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' 

print("Bắt đầu khởi tạo hệ thống tìm kiếm text-vs-text...")

# 1. Tải model Sentence Transformer
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"✅ Model '{MODEL_NAME}' đã được tải trên {device.upper()}.")
except Exception as e:
    print(f"LỖI: Không thể tải model Sentence Transformer: {e}")
    exit(1)

# 2. Tải và xử lý database captions
try:
    with open(CAPTIONS_DB_FILE, "r", encoding='utf-8') as f:
        database_captions = json.load(f)
    
    # Tách riêng caption và đường dẫn ảnh để xử lý
    corpus_captions = [item['caption'] for item in database_captions]
    corpus_image_paths = [item['image_path'] for item in database_captions]
    print(f"✅ Đã tải {len(corpus_captions)} captions từ '{CAPTIONS_DB_FILE}'.")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file '{CAPTIONS_DB_FILE}'. Vui lòng chạy script 'generate_database_captions.py' trước.")
    exit(1)

# 3. Mã hóa toàn bộ corpus captions (chỉ làm 1 lần)
# Đây là bước quan trọng để tìm kiếm diễn ra nhanh chóng
print("Đang mã hóa toàn bộ captions trong database. Việc này có thể mất vài phút...")
corpus_embeddings = model.encode(corpus_captions, convert_to_tensor=True, show_progress_bar=True)
print("✅ Mã hóa hoàn tất.")

def search(query: str, top_k: int = 8):
    """
    Hàm tìm kiếm caption tương đồng nhất với câu truy vấn.
    """
    print(f"\n🔍 Bắt đầu tìm kiếm cho truy vấn: '{query}'")
    
    # Mã hóa câu truy vấn
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Sử dụng hàm semantic_search của thư viện để tìm kiếm hiệu quả
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    
    # hits là một list các kết quả, ta chỉ lấy kết quả của query đầu tiên (và duy nhất)
    hits = hits[0] 
    
    results = []
    print("--- Kết quả tìm kiếm ---")
    for hit in hits:
        # Lấy caption và đường dẫn ảnh tương ứng với kết quả
        result_caption = corpus_captions[hit['corpus_id']]
        result_path = corpus_image_paths[hit['corpus_id']]
        score = hit['score']
        print(f"Score: {score:.4f} - Path: {result_path} - Caption: '{result_caption}'")
        results.append({
            "path": result_path,
            "score": score
        })
    return results

# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    # Ví dụ các câu truy vấn
    search("two cats sitting on a sofa")
    search("a red car on the street")
    search("people playing on the beach")