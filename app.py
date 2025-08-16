# app.py
import streamlit as st
from PIL import Image
import numpy as np
import pickle
from feature_extractor import FeatureExtractorBlip2

# --- Cấu hình trang ---
st.set_page_config(page_title="Hệ thống Truy vấn Đa phương tiện", layout="wide")

# --- Tải dữ liệu và model ---

@st.cache_resource
def load_model():
    """Tải model một lần duy nhất."""
    return FeatureExtractorBlip2()

@st.cache_data
def load_indexed_data(file_path):
    """Tải dữ liệu đã được lập chỉ mục."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["paths"], data["features"]

# Tải model và dữ liệu
fe = load_model()
img_paths, features_array = load_indexed_data(fe.FEATURES_FILE)

# --- Giao diện người dùng ---
st.title("Hệ thống Truy vấn Đa phương tiện 🖼️📝")
st.write("Tìm kiếm hình ảnh tương tự trong bộ dữ liệu bằng cách sử dụng hình ảnh hoặc văn bản mô tả.")

# Tạo 2 tab cho 2 loại query
tab1, tab2 = st.tabs(["🔍 Tìm kiếm bằng Hình ảnh", "📝 Tìm kiếm bằng Văn bản"])

# --- Tab 1: Tìm kiếm bằng Hình ảnh ---
with tab1:
    st.header("Tải lên một ảnh để tìm kiếm")
    uploaded_file = st.file_uploader("Chọn một tệp ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        query_image = Image.open(uploaded_file)
        st.image(query_image, caption="Ảnh truy vấn", width=200)

        if st.button("Tìm kiếm ảnh tương tự"):
            with st.spinner("Đang trích xuất đặc trưng và tìm kiếm..."):
                # Trích xuất đặc trưng của ảnh truy vấn
                query_features = fe.extract_image_features(query_image)

                # --- ĐÃ SỬA ---
                # Tính độ tương đồng bằng tích vô hướng (dot product)
                similarities = np.dot(features_array, query_features)

                # Lấy ra 12 kết quả có độ tương đồng cao nhất (sắp xếp giảm dần)
                result_indices = np.argsort(-similarities)[:12]
                # --- KẾT THÚC SỬA ---

                st.header("Kết quả tìm kiếm:")
                # Hiển thị kết quả theo lưới
                cols = st.columns(6)
                for i, idx in enumerate(result_indices):
                    with cols[i % 6]:
                        st.image(img_paths[idx], use_column_width=True)

# --- Tab 2: Tìm kiếm bằng Văn bản ---
with tab2:
    st.header("Nhập mô tả văn bản để tìm kiếm")
    query_text = st.text_input("Ví dụ: 'a red car on the street', 'two cats sitting on a sofa'")

    if st.button("Tìm kiếm bằng văn bản"):
        if query_text:
            with st.spinner("Đang mã hóa văn bản và tìm kiếm..."):
                # Trích xuất đặc trưng của văn bản truy vấn
                query_features = fe.extract_text_features(query_text)

                # --- ĐÃ SỬA ---
                # Tính độ tương đồng bằng tích vô hướng (dot product)
                similarities = np.dot(features_array, query_features)

                # Lấy ra 12 kết quả có độ tương đồng cao nhất (sắp xếp giảm dần)
                result_indices = np.argsort(-similarities)[:12]
                # --- KẾT THÚC SỬA ---

                st.header("Kết quả tìm kiếm:")
                # Hiển thị kết quả theo lưới
                cols = st.columns(6)
                for i, idx in enumerate(result_indices):
                    with cols[i % 6]:
                        st.image(img_paths[idx], use_column_width=True)
        else:
            st.warning("Vui lòng nhập mô tả văn bản.")


# --- Hướng dẫn sử dụng bên sidebar ---
st.sidebar.title("Hướng dẫn")
st.sidebar.markdown("""
1.  **Chuẩn bị Dataset:**
    -   Tạo một thư mục có tên `dataset`.
    -   Chép tất cả hình ảnh của bạn vào thư mục này.

2.  **Lập chỉ mục:**
    -   Chạy file `index.py` từ terminal:
        ```bash
        python index.py
        ```
    -   File `features_siglip.pkl` sẽ được tạo ra.

3.  **Chạy ứng dụng:**
    -   Chạy file `app.py` từ terminal:
        ```bash
        streamlit run app.py
        ```
""")