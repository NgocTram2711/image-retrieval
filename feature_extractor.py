# feature_extractor_CLIP.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import SiglipProcessor, SiglipModel

class FeatureExtractorClip:
    FEATURES_FILE = "features_clip.pkl"

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        model_name = "openai/clip-vit-large-patch14"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        print(f"✅ Model CLIP has been loaded and is running on device: {self.device.upper()}")

    def extract_image_features(self, img_path_or_pil):
        if isinstance(img_path_or_pil, str):
            image = Image.open(img_path_or_pil).convert("RGB")
        else:
            image = img_path_or_pil.convert("RGB")

        # SỬA LỖI Ở ĐÂY: Bỏ `torch.float16` ra khỏi hàm .to()
        # inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16) # Dòng cũ
        inputs = self.processor(images=image, return_tensors="pt").to(self.device) # Dòng mới

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()

    def extract_text_features(self, text):
        # SỬA LỖI Ở ĐÂY: Bỏ `torch.float16` ra khỏi hàm .to()
        # inputs = self.processor(text=text, return_tensors="pt").to(self.device, torch.float16) # Dòng cũ
        inputs = self.processor(text=text, return_tensors="pt").to(self.device) # Dòng mới
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()
    

class FeatureExtractorSiglip2:
    FEATURES_FILE = "features_siglip2.pkl"

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        model_name = "google/siglip2-large-patch16-512"
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        print(f"✅ Model SigLIP '{model_name.split('/')[-1]}' has been loaded and is running on device: {self.device.upper()}")

    def extract_image_features(self, img_path_or_pil):
        if isinstance(img_path_or_pil, str):
            image = Image.open(img_path_or_pil).convert("RGB")
        else:
            image = img_path_or_pil.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # --- BỎ ĐI DÒNG CHUẨN HÓA NÀY ---
        # image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()

    def extract_text_features(self, text):
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # --- BỎ ĐI DÒNG CHUẨN HÓA NÀY ---
        # text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()