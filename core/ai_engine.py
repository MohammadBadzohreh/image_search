# core/ai_engine.py
import streamlit as st
from transformers import (
    AutoProcessor, SiglipModel, AutoModel, 
    BlipProcessor, BlipForConditionalGeneration,
    CLIPModel, CLIPProcessor
)
from PIL import Image
import torch
import config
# کتابخانه برای CLIPA
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
            
        # 2. Jina CLIP
        elif cfg["type"] == "jina":
            model = AutoModel.from_pretrained(cfg["model_id"], trust_remote_code=True).to(config.DEVICE)
            return model, None, "jina"
            
        # 3. OpenCLIP (CLIPA)
        elif cfg["type"] == "open_clip":
            model, _, preprocess = open_clip.create_model_and_transforms(cfg["model_id"], device=config.DEVICE)
            tokenizer = open_clip.get_tokenizer(cfg["model_id"])
            return model, (preprocess, tokenizer), "open_clip"
            
        # 4. Llama Nemo (Multimodal) 👇
        elif cfg["type"] == "llama_nemo":
            # این مدل سنگین است (3B)، اگر GPU دارید بهتر است با float16 لود شود
            dtype = torch.float16 if config.DEVICE == "cuda" else torch.float32
            model = AutoModel.from_pretrained(
                cfg["model_id"], 
                trust_remote_code=True, 
                torch_dtype=torch.float32  # 👈 این خط مشکل را حل می‌کند
                # torch_dtype=dtype
            ).to(config.DEVICE)
            # این مدل خودش پروسسور داخلی دارد
            return model, None, "llama_nemo"

    def get_embedding(self, model_key, image=None, text=None):
        loaded_data = self.load_embedding_model(model_key)
        
        if len(loaded_data) == 3:
            model, processor, model_type = loaded_data
        else:
            model, processor, model_type = loaded_data[0], None, "jina"

        vector = None
        
        with torch.no_grad():
            
            # --- منطق Llama Nemo (Multimodal) 👇 ---
            if model_type == "llama_nemo":
                embeddings = None
                
                if image:
                    # لود کردن تصویر
                    if isinstance(image, str): 
                        pil_image = Image.open(image).convert("RGB")
                    else: 
                        pil_image = image.convert("RGB")
                    
                    # این مدل متد اختصاصی forward_passages دارد که عکس می‌گیرد
                    # خروجی: [batch, num_tokens, dim]
                    output = model.forward_passages([pil_image], batch_size=1)
                    embeddings = output
                    
                elif text:
                    # این مدل متد اختصاصی forward_queries دارد برای متن
                    output = model.forward_queries([text], batch_size=1)
                    embeddings = output

                # تبدیل به تک‌بردار (Mean Pooling)
                # چون خروجی ColBERT چند برداری است، برای Milvus باید میانگین بگیریم
                if embeddings is not None:
                    # embeddings shape: [1, seq_len, 3072]
                    pooled = embeddings.mean(dim=1) # میانگین روی توکن‌ها
                    
                    # نرمال‌سازی (L2 Norm)
                    pooled = pooled / pooled.norm(p=2, dim=-1, keepdim=True)
                    
                    # تبدیل به float32 (اگر مدل fp16 باشد، نامپای باید 32 باشد)
                    vector = pooled[0].float().cpu().numpy()

            # --- منطق SigLIP ---
            elif model_type == "siglip":
                if image:
                    if isinstance(image, str): image = Image.open(image).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
                    features = model.get_image_features(**inputs)
                elif text:
                    inputs = processor(text=[text], return_tensors="pt", padding="max_length", max_length=64).to(config.DEVICE)
                    features = model.get_text_features(**inputs)
                
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                vector = features[0].cpu().numpy()

            # --- منطق Jina CLIP ---
            elif model_type == "jina":
                if image:
                    if isinstance(image, str): image = Image.open(image).convert("RGB")
                    vector = model.encode_image(image) 
                elif text:
                    vector = model.encode_text(text)
                
                if isinstance(vector, torch.Tensor): vector = vector.cpu().numpy()
                if vector.ndim > 1: vector = vector[0]

            # --- منطق OpenCLIP (CLIPA) ---
            elif model_type == "open_clip":
                preprocess, tokenizer = processor
                if image:
                    if isinstance(image, str): image = Image.open(image).convert("RGB")
                    image_tensor = preprocess(image).unsqueeze(0).to(config.DEVICE)
                    features = model.encode_image(image_tensor)
                elif text:
                    text_tokens = tokenizer([text]).to(config.DEVICE)
                    features = model.encode_text(text_tokens)
                
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
        try:
            raw_image = Image.open(image_path).convert('RGB')
            inputs = processor(raw_image, return_tensors="pt").to(config.DEVICE)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)
            return processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            return "error in ai engine"