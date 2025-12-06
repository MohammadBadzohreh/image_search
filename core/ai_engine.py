# core/ai_engine.py
import streamlit as st
from transformers import AutoProcessor, SiglipModel
from PIL import Image
import torch
import config

class AIEngine:
    @staticmethod
    @st.cache_resource
    def load_model():
        print(f"🔄 Loading AI Model on {config.DEVICE}...")
        model = SiglipModel.from_pretrained(config.MODEL_NAME).to(config.DEVICE)
        processor = AutoProcessor.from_pretrained(config.MODEL_NAME)
        return model, processor

    def get_embedding(self, image=None, text=None):
        model, processor = self.load_model()
        
        inputs = None
        if image:
            # تبدیل عکس به تنسور
            if isinstance(image, str): # اگر آدرس فایل بود
                image = Image.open(image).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
            
            with torch.no_grad():
                features = model.get_image_features(**inputs)

        elif text:
            # تبدیل متن به تنسور
            inputs = processor(text=[text], return_tensors="pt", padding="max_length", max_length=64).to(config.DEVICE)
            with torch.no_grad():
                features = model.get_text_features(**inputs)
        
        # نرمال‌سازی بردار (خیلی مهم برای Cosine Similarity)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features[0].cpu().numpy()