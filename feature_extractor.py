# feature_extractor.py
import torch
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel, 
    SiglipProcessor, SiglipModel,
    AutoProcessor, Blip2Model,
    AutoImageProcessor, Dinov2Model
)

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
    
class FeatureExtractorBlip2:
    """
    Trình trích xuất đặc trưng sử dụng mô hình BLIP-2 của Salesforce.
    Sử dụng Q-Former để đảm bảo đặc trưng ảnh và văn bản có cùng kích thước (768).
    """
    FEATURES_FILE = "features_blip2.pkl"

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        model_name = "Salesforce/blip2-opt-2.7b"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2Model.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        # Create a dummy image (1x3x224x224) for text feature extraction
        self.dummy_image = torch.zeros(1, 3, 224, 224, dtype=torch.float16).to(self.device)
        # Create dummy input_ids (empty text) for image feature extraction
        self.dummy_input_ids = torch.zeros(1, 1, dtype=torch.long).to(self.device)
        print(f"✅ Model BLIP-2 loaded. Running on: {self.device.upper()}")

    def _normalize(self, features):
        """Hàm trợ giúp chuẩn hóa L2."""
        features = features / torch.norm(features, p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def extract_image_features(self, img_path_or_pil):
        """Trích xuất đặc trưng ảnh bằng Q-Former."""
        image = Image.open(img_path_or_pil).convert("RGB") if isinstance(img_path_or_pil, str) else img_path_or_pil.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        # Add dummy input_ids to satisfy Blip2Model's forward pass
        inputs["input_ids"] = self.dummy_input_ids

        with torch.no_grad():
            # Full forward pass to get Q-Former output
            outputs = self.model(**inputs, return_dict=True)
            # Extract image features from Q-Former pooler output
            image_features = outputs.qformer_outputs.pooler_output

        return self._normalize(image_features)

    def extract_text_features(self, text):
        """Trích xuất đặc trưng văn bản bằng Q-Former với dummy image."""
        inputs = self.processor(text=text, return_tensors="pt").to(self.device, torch.float16)
        # Add dummy image input to satisfy Blip2Model's forward pass
        inputs["pixel_values"] = self.dummy_image

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            # Extract text features from Q-Former pooler output
            text_features = outputs.qformer_outputs.pooler_output

        return self._normalize(text_features)
                            
# --- CLASS MỚI CHO DINOv2 ---
class FeatureExtractorDinoV2:
    """
    Trình trích xuất đặc trưng chỉ dành cho HÌNH ẢNH sử dụng DINOv2 của Meta.
    Model này cực kỳ mạnh cho tìm kiếm ảnh-bằng-ảnh (image-to-image).
    Lưu ý: KHÔNG hỗ trợ trích xuất đặc trưng văn bản.
    """
    FEATURES_FILE = "features_dinov2.pkl"

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        model_name = "facebook/dinov2-large"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        print(f"✅ Model DINOv2 loaded. Running on: {self.device.upper()}")

    def extract_image_features(self, img_path_or_pil):
        """Trích xuất vector đặc trưng từ một ảnh."""
        image = Image.open(img_path_or_pil).convert("RGB") if isinstance(img_path_or_pil, str) else img_path_or_pil.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Lấy đặc trưng từ token [CLS] cuối cùng
            image_features = outputs.last_hidden_state
            
        # Lấy pooler output và chuẩn hóa L2
        features = image_features[:, 0]
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def extract_text_features(self, text):
        """Phương thức này không được hỗ trợ bởi DINOv2."""
        raise NotImplementedError("DINOv2 là một mô hình chỉ dành cho thị giác và không thể trích xuất đặc trưng văn bản.")
