# feature_extractor.py
import torch
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel, 
    SiglipProcessor, SiglipModel,
    AutoProcessor, Blip2ForConditionalGeneration,
    AutoImageProcessor, Dinov2Model
)
import os

class FeatureExtractorClip:
    """
    Sử dụng mô hình CLIP đã được fine-tune.
    """
    # ĐÃ SỬA: Tên file features đồng bộ với evaluate.py
    FEATURES_FILE = "features_clip.pkl" 

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # ĐÃ SỬA: Tải model đã fine-tune từ thư mục cục bộ
        model_name = "./clip_finetuned_coco" 
        
        if not os.path.exists(model_name):
            print(f"LỖI: Không tìm thấy thư mục model đã fine-tune '{model_name}'.")
            print("Vui lòng chạy script fine-tuning trước.")
            exit(1)

        try:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
            print(f"✅ Model CLIP đã fine-tune được tải từ '{model_name}'. Đang chạy trên: {self.device.upper()}")
        except Exception as e:
            print(f"LỖI: Không thể tải mô hình từ {model_name}: {e}")
            exit(1)

    def _extract_features(self, inputs, feature_type):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_image_features(**inputs) if feature_type == "image" else self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def extract_image_features(self, img_path_or_pil):
        image = Image.open(img_path_or_pil).convert("RGB") if isinstance(img_path_or_pil, str) else img_path_or_pil.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return self._extract_features(inputs, "image")

    def extract_text_features(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        return self._extract_features(inputs, "text")

class FeatureExtractorSiglip2:
    """
    Sử dụng mô hình SigLIP đã được fine-tune.
    """
    # ĐÃ SỬA: Tên file features đồng bộ với evaluate.py
    FEATURES_FILE = "features_siglip2.pkl"

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # ĐÃ SỬA: Tải model đã fine-tune từ thư mục cục bộ
        model_name = "./siglip_finetuned_coco"

        if not os.path.exists(model_name):
            print(f"LỖI: Không tìm thấy thư mục model đã fine-tune '{model_name}'.")
            print("Vui lòng chạy script fine-tuning trước.")
            exit(1)
            
        try:
            self.processor = SiglipProcessor.from_pretrained(model_name)
            self.model = SiglipModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
            print(f"✅ Model SigLIP đã fine-tune được tải từ '{model_name}'. Đang chạy trên: {self.device.upper()}")
        except Exception as e:
            print(f"LỖI: Không thể tải mô hình từ {model_name}: {e}")
            exit(1)

    def _extract_features(self, inputs, feature_type):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_image_features(**inputs) if feature_type == "image" else self.model.get_text_features(**inputs)
        return features.cpu().numpy().flatten()

    def extract_image_features(self, img_path_or_pil):
        image = Image.open(img_path_or_pil).convert("RGB") if isinstance(img_path_or_pil, str) else img_path_or_pil.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        # Cast inputs to float16 to match model dtype
        inputs = {k: v.to(dtype=torch.float16) for k, v in inputs.items()}
        return self._extract_features(inputs, "image")

    def extract_text_features(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        return self._extract_features(inputs, "text")

class FeatureExtractorBlip2:
    """
    Sử dụng mô hình BLIP-2 gốc từ Salesforce (không fine-tune).
    """
    FEATURES_FILE = "features_blip2.pkl"

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        model_name = "Salesforce/blip2-opt-6.7b-coco"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.dummy_image = torch.zeros(1, 3, 224, 224, dtype=torch.float16).to(self.device)
        self.dummy_input_ids = torch.zeros(1, 1, dtype=torch.long).to(self.device)
        print(f"✅ Model BLIP-2 gốc được tải. Đang chạy trên: {self.device.upper()}")

    def _normalize(self, features):
        features = features / torch.norm(features, p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def extract_image_features(self, img_path_or_pil):
        image = Image.open(img_path_or_pil).convert("RGB") if isinstance(img_path_or_pil, str) else img_path_or_pil.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        inputs["input_ids"] = self.dummy_input_ids
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            image_features = outputs.qformer_outputs.pooler_output
        return self._normalize(image_features)

    def extract_text_features(self, text):
        inputs = self.processor(text=text, return_tensors="pt").to(self.device, torch.float16)
        inputs["pixel_values"] = self.dummy_image
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            text_features = outputs.qformer_outputs.pooler_output
        return self._normalize(text_features)

    def generate_caption(self, image):
        """Sinh caption cho ảnh bằng Blip2."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption.strip()

# Lớp FeatureExtractorDinoV2 giữ nguyên vì không liên quan đến flow này
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
        image = Image.open(img_path_or_pil).convert("RGB") if isinstance(img_path_or_pil, str) else img_path_or_pil.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.last_hidden_state
        features = image_features[:, 0]
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def extract_text_features(self, text):
        raise NotImplementedError("DINOv2 là một mô hình chỉ dành cho thị giác và không thể trích xuất đặc trưng văn bản.")