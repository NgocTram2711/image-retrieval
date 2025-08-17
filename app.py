# app.py
import streamlit as st
from PIL import Image
import numpy as np
import pickle
import json
from sentence_transformers import SentenceTransformer, util

# Import các class extractor từ file feature_extractor.py
from feature_extractor import FeatureExtractorClip, FeatureExtractorSiglip2, FeatureExtractorBlip2

# --- Cấu hình trang ---
st.set_page_config(page_title="Hệ thống Truy vấn Hình ảnh", layout="wide")

# --- Dữ liệu hiệu năng của các model ---
PERFORMANCE_METRICS = {
    "CLIP (Fine-tuned)": {
        "text_to_image": {"mAP": 0.221, "mRecall@10": 0.490},
        "image_to_image": {"mAP": 0.996, "mRecall@10": 1.000},
    },
    "SigLIP (Fine-tuned)": {
        "text_to_image": {"mAP": 0.002, "mRecall@10": 0.002},
        "image_to_image": {"mAP": 0.997, "mRecall@10": 1.000},
    },
    "BLIP-2 (Caption Search)": {
        "text_to_image": {"mAP": 0.326, "mRecall@10": 0.584},
        "image_to_image": {"mAP": 1.000, "mRecall@10": 1.000},
    }
}

# --- Tải dữ liệu và model (sử dụng cache của Streamlit) ---

@st.cache_resource
def load_all_models():
    """Tải tất cả các model cần thiết một lần duy nhất."""
    models = {
        "CLIP (Fine-tuned)": FeatureExtractorClip(),
        "SigLIP (Fine-tuned)": FeatureExtractorSiglip2(),
        "BLIP-2 (Caption Search)": FeatureExtractorBlip2(),
        "SentenceTransformer": SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    }
    return models

@st.cache_data
def load_indexed_data(model_name):
    """Tải dữ liệu đã được lập chỉ mục cho một model cụ thể."""
    if "CLIP" in model_name:
        file_path = "features_clip.pkl"
    elif "SigLIP" in model_name:
        file_path = "features_siglip2.pkl"
    else: # BLIP-2
        file_path = "features_blip2.pkl"
    
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data["paths"], data["features"]
    except FileNotFoundError:
        st.error(f"LỖI: Không tìm thấy file features '{file_path}'. Vui lòng chạy 'index.py --model ...' cho model này.")
        return None, None

@st.cache_data
def load_and_encode_captions(_st_model):
    """Tải và mã hóa captions của database cho BLIP-2."""
    try:
        with open("db_captions.json", "r", encoding='utf-8') as f:
            db_captions_data = json.load(f)
        
        # Sắp xếp captions theo đúng thứ tự của image paths
        img_paths, _ = load_indexed_data("BLIP-2 (Caption Search)")
        path_to_caption = {item['image_path']: item['caption'] for item in db_captions_data}
        ordered_captions = [path_to_caption.get(p, "") for p in img_paths]

        embeddings = _st_model.encode(ordered_captions, convert_to_tensor=True, show_progress_bar=True)
        return ordered_captions, embeddings
    except FileNotFoundError:
        st.error("LỖI: Không tìm thấy file 'db_captions.json'. Vui lòng chạy 'generate_database_captions.py' trước.")
        return None, None

# Tải tất cả model khi khởi động
models = load_all_models()

# --- Giao diện người dùng ---

# Sidebar để chọn model
st.sidebar.title("⚙️ Tùy chọn")
selected_model_name = st.sidebar.selectbox(
    "Chọn mô hình để tìm kiếm:",
    list(PERFORMANCE_METRICS.keys())
)

# Hiển thị hiệu năng của model đã chọn
st.sidebar.header("Hiệu năng Model")
metrics = PERFORMANCE_METRICS[selected_model_name]
st.sidebar.markdown("**Text-to-Image:**")
st.sidebar.text(f"  - mAP: {metrics['text_to_image']['mAP']:.3f}")
st.sidebar.text(f"  - mRecall@10: {metrics['text_to_image']['mRecall@10']:.3f}")
st.sidebar.markdown("**Image-to-Image:**")
st.sidebar.text(f"  - mAP: {metrics['image_to_image']['mAP']:.3f}")
st.sidebar.text(f"  - mRecall@10: {metrics['image_to_image']['mRecall@10']:.3f}")


# Tiêu đề chính
st.title("Hệ thống Truy vấn Hình ảnh Đa phương tiện 🖼️📝")
st.write(f"Hiện đang sử dụng model: **{selected_model_name}**")

# Tải dữ liệu tương ứng với model đã chọn
img_paths, features_array = load_indexed_data(selected_model_name)

# Nếu là BLIP-2, tải thêm dữ liệu caption
db_captions, caption_embeddings = (None, None)
if "BLIP-2" in selected_model_name:
    db_captions, caption_embeddings = load_and_encode_captions(models["SentenceTransformer"])

# Tạo 2 tab cho 2 loại query
tab1, tab2 = st.tabs(["🔍 Tìm kiếm bằng Hình ảnh", "📝 Tìm kiếm bằng Văn bản"])

# --- Tab 1: Tìm kiếm bằng Hình ảnh ---
with tab1:
    st.header("Tải lên một ảnh để tìm kiếm")
    uploaded_file = st.file_uploader("Chọn một tệp ảnh", type=["jpg", "jpeg", "png"], key="image_uploader")

    if uploaded_file and img_paths:
        query_image = Image.open(uploaded_file)
        st.image(query_image, caption="Ảnh truy vấn", width=200)

        if st.button("Tìm kiếm ảnh tương tự", key="search_image"):
            with st.spinner("Đang trích xuất đặc trưng và tìm kiếm..."):
                fe = models[selected_model_name]
                query_features = fe.extract_image_features(query_image)
                
                # Logic tìm kiếm Image-to-Image là giống nhau cho cả 3 model
                similarities = np.dot(features_array, query_features)
                result_indices = np.argsort(-similarities)[:12]

                st.header("Kết quả tìm kiếm:")
                cols = st.columns(6)
                for i, idx in enumerate(result_indices):
                    with cols[i % 6]:
                        st.image(img_paths[idx], use_column_width=True)

# --- Tab 2: Tìm kiếm bằng Văn bản ---
with tab2:
    st.header("Nhập mô tả văn bản để tìm kiếm")
    query_text = st.text_input("Ví dụ: 'a red car on the street', 'two cats sitting on a sofa'")

    if st.button("Tìm kiếm bằng văn bản", key="search_text"):
        if query_text and img_paths:
            with st.spinner("Đang mã hóa văn bản và tìm kiếm..."):
                
                # --- LOGIC TÌM KIẾM PHÂN NHÁNH ---
                if "BLIP-2" in selected_model_name:
                    # Phương pháp Text-vs-Text cho BLIP-2
                    if caption_embeddings is not None:
                        st_model = models["SentenceTransformer"]
                        query_embedding = st_model.encode(query_text, convert_to_tensor=True)
                        hits = util.semantic_search(query_embedding, caption_embeddings, top_k=12)[0]
                        result_indices = [hit['corpus_id'] for hit in hits]
                    else:
                        st.error("Không thể tải dữ liệu caption cho BLIP-2.")
                        result_indices = []
                else:
                    # Phương pháp truyền thống cho CLIP và SigLIP
                    fe = models[selected_model_name]
                    query_features = fe.extract_text_features(query_text)
                    similarities = np.dot(features_array, query_features)
                    result_indices = np.argsort(-similarities)[:12]

                st.header("Kết quả tìm kiếm:")
                cols = st.columns(6)
                for i, idx in enumerate(result_indices):
                    with cols[i % 6]:
                        st.image(img_paths[idx], use_column_width=True)
        else:
            st.warning("Vui lòng nhập mô tả văn bản.")
