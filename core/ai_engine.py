# core/ai_engine.py
import streamlit as st
from transformers import AutoProcessor, SiglipModel, AutoModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import config

class AIEngine:
    
    @staticmethod
    @st.cache_resource
    def load_embedding_model(model_key):
        """
        لود کردن مدل بر اساس انتخاب کاربر
        """
        cfg = config.MODELS_CONFIG[model_key]
        print(f"🔄 Loading {model_key} ({cfg['model_id']}) on {config.DEVICE}...")
        
        if cfg["type"] == "siglip":
            model = SiglipModel.from_pretrained(cfg["model_id"]).to(config.DEVICE)
            processor = AutoProcessor.from_pretrained(cfg["model_id"])
            return model, processor, "siglip"
            
        elif cfg["type"] == "jina":
            # هر دو نسخه Jina v1 و Jina v2 با این روش لود می‌شوند
            model = AutoModel.from_pretrained(cfg["model_id"], trust_remote_code=True).to(config.DEVICE)
            return model, None, "jina"

    def get_embedding(self, model_key, image=None, text=None):
        # 1. لود مدل مناسب
        loaded_data = self.load_embedding_model(model_key)
        
        if len(loaded_data) == 3:
            model, processor, model_type = loaded_data
        else:
            model, processor, model_type = loaded_data[0], None, "jina"

        vector = None
        
        # --- منطق SigLIP ---
        if model_type == "siglip":
            if image:
                if isinstance(image, str): image = Image.open(image).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
                with torch.no_grad():
                    features = model.get_image_features(**inputs)
            elif text:
                inputs = processor(text=[text], return_tensors="pt", padding="max_length", max_length=64).to(config.DEVICE)
                with torch.no_grad():
                    features = model.get_text_features(**inputs)
            
            # نرمال‌سازی برای SigLIP
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            vector = features[0].cpu().numpy()

        # --- منطق Jina CLIP (v1 & v2) ---
        elif model_type == "jina":
            with torch.no_grad():
                if image:
                    if isinstance(image, str): image = Image.open(image).convert("RGB")
                    vector = model.encode_image(image) 
                elif text:
                    # Jina v2 تا 8k توکن ساپورت می‌کند
                    vector = model.encode_text(text)
            
            if isinstance(vector, torch.Tensor):
                vector = vector.cpu().numpy()
            
            if vector.ndim > 1:
                vector = vector[0]

        return vector

    # --- بخش کپشن (BLIP) ---
    @staticmethod
    @st.cache_resource
    def load_caption_model():
        print(f"🔄 Loading BLIP on {config.DEVICE}...")
        processor = BlipProcessor.from_pretrained(config.CAPTION_MODEL)
        model = BlipForConditionalGeneration.from_pretrained(config.CAPTION_MODEL).to(config.DEVICE)
        return model, processor

    def generate_caption(self, image_path):
        model, processor = self.load_caption_model()
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to(config.DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0], skip_special_tokens=True)