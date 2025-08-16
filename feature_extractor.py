# feature_extractor.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, SiglipProcessor, SiglipModel

class FeatureExtractorClip:
    """
    Trình trích xuất đặc trưng sử dụng mô hình CLIP của OpenAI.
    Lưu ý: Các đặc trưng được chuẩn hóa L2.
    """
    FEATURES_FILE = "features_clip.pkl"

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        model_name = "openai/clip-vit-large-patch14"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        # Tải model với độ chính xác float16 để tăng tốc độ trên GPU tương thích
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        print(f"✅ Model CLIP loaded. Running on: {self.device.upper()}")

    def _extract_features(self, inputs, feature_type):
        """Hàm trợ giúp để trích xuất và chuẩn hóa đặc trưng."""
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            if feature_type == "image":
                features = self.model.get_image_features(**inputs)
            else:
                features = self.model.get_text_features(**inputs)
        
        # CLIP yêu cầu chuẩn hóa L2
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def extract_image_features(self, img_path_or_pil):
        """Trích xuất vector đặc trưng từ một ảnh."""
        image = Image.open(img_path_or_pil).convert("RGB") if isinstance(img_path_or_pil, str) else img_path_or_pil.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return self._extract_features(inputs, "image")

    def extract_text_features(self, text):
        """Trích xuất vector đặc trưng từ một chuỗi văn bản."""
        inputs = self.processor(text=text, return_tensors="pt")
        return self._extract_features(inputs, "text")

class FeatureExtractorSiglip2:
    """
    Trình trích xuất đặc trưng sử dụng mô hình SigLIP của Google.
    Lưu ý: Các đặc trưng KHÔNG được chuẩn hóa. Sử dụng tích vô hướng để đo độ tương đồng.
    """
    FEATURES_FILE = "features_siglip2_train.pkl"

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        model_name = "google/siglip2-large-patch16-512"
        self.processor = SiglipProcessor.from_pretrained(model_name)
        # Tải model với độ chính xác float16 để tăng tốc độ trên GPU tương thích
        self.model = SiglipModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        print(f"✅ Model SigLIP-v2 loaded. Running on: {self.device.upper()}")

    def _extract_features(self, inputs, feature_type):
        """Hàm trợ giúp để trích xuất đặc trưng."""
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            if feature_type == "image":
                features = self.model.get_image_features(**inputs)
            else:
                features = self.model.get_text_features(**inputs)
        # SigLIP không cần chuẩn hóa
        return features.cpu().numpy().flatten()

    def extract_image_features(self, img_path_or_pil):
        """Trích xuất vector đặc trưng từ một ảnh."""
        image = Image.open(img_path_or_pil).convert("RGB") if isinstance(img_path_or_pil, str) else img_path_or_pil.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return self._extract_features(inputs, "image")

    def extract_text_features(self, text):
        """Trích xuất vector đặc trưng từ một chuỗi văn bản."""
        inputs = self.processor(text=text, return_tensors="pt")
        return self._extract_features(inputs, "text")