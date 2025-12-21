# core/ai_engine.py
import streamlit as st
from transformers import AutoProcessor, SiglipModel, AutoModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import config
# 👇 کتابخانه جدید
import open_clip

class AIEngine:
    
    @staticmethod
    @st.cache_resource
    def load_embedding_model(model_key):
        cfg = config.MODELS_CONFIG[model_key]
        print(f"🔄 Loading {model_key} ({cfg['model_id']}) on {config.DEVICE}...")
        
        # 1. SigLIP
        if cfg["type"] == "siglip":
            model = SiglipModel.from_pretrained(cfg["model_id"]).to(config.DEVICE)
            processor = AutoProcessor.from_pretrained(cfg["model_id"])
            return model, processor, "siglip"
            
        # 2. Jina CLIP (v1 & v2)
        elif cfg["type"] == "jina":
            model = AutoModel.from_pretrained(cfg["model_id"], trust_remote_code=True).to(config.DEVICE)
            return model, None, "jina"
            
        # 3. OpenCLIP (برای CLIPA-v2 شما) 👇
        elif cfg["type"] == "open_clip":
            # این دستور دقیقاً مدل را از لینک HF شما می‌خواند
            model, _, preprocess = open_clip.create_model_and_transforms(cfg["model_id"], device=config.DEVICE)
            tokenizer = open_clip.get_tokenizer(cfg["model_id"])
            return model, (preprocess, tokenizer), "open_clip"

    def get_embedding(self, model_key, image=None, text=None):
        loaded_data = self.load_embedding_model(model_key)
        
        if len(loaded_data) == 3:
            model, processor, model_type = loaded_data
        else:
            model, processor, model_type = loaded_data[0], None, "jina"

        vector = None
        
        # --- SigLIP ---
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
            
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            vector = features[0].cpu().numpy()

        # --- Jina CLIP ---
        elif model_type == "jina":
            with torch.no_grad():
                if image:
                    if isinstance(image, str): image = Image.open(image).convert("RGB")
                    vector = model.encode_image(image) 
                elif text:
                    vector = model.encode_text(text)
            
            if isinstance(vector, torch.Tensor): vector = vector.cpu().numpy()
            if vector.ndim > 1: vector = vector[0]

        # --- OpenCLIP (CLIPA-v2) 👇 ---
        elif model_type == "open_clip":
            preprocess, tokenizer = processor # باز کردن پکیج
            
            with torch.no_grad():
                if image:
                    if isinstance(image, str): image = Image.open(image).convert("RGB")
                    # پیش‌پردازش مخصوص OpenCLIP (تغییر سایز به 336)
                    image_tensor = preprocess(image).unsqueeze(0).to(config.DEVICE)
                    features = model.encode_image(image_tensor)
                elif text:
                    # توکنایزر مخصوص OpenCLIP
                    text_tokens = tokenizer([text]).to(config.DEVICE)
                    features = model.encode_text(text_tokens)
            
            # نرمال‌سازی بسیار مهم است
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            vector = features[0].cpu().numpy()

        return vector

    # --- BLIP (Caption) ---
    @staticmethod
    @st.cache_resource
    def load_caption_model():
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