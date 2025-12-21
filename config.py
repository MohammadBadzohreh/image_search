# config.py
import torch

# --- تنظیمات عمومی ---
MILVUS_URI = "http://milvus-standalone:19530" 
IMAGE_STORAGE_PATH = "/home/jovyan/work/benchmark/data/flickr30k/Images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- تنظیمات مدل کپشن ---
CAPTION_MODEL = "Salesforce/blip-image-captioning-base" 

# --- تنظیمات مدل‌های هوش مصنوعی (۳ مدل همزمان) ---
MODELS_CONFIG = {
    "SigLIP": {
        "model_id": "google/siglip-so400m-patch14-384",
        "collection_name": "siglip_gallery_v3_captioned",
        "dimension": 1152,  # ابعاد سیگ‌لیپ
        "type": "siglip"
    },
    "Jina CLIP v1": {
        "model_id": "jinaai/jina-clip-v1", 
        "collection_name": "jina_clip_v1_embedding", # کالکشن نسخه ۱
        "dimension": 768,   # ابعاد نسخه ۱
        "type": "jina"
    },
    "Jina CLIP v2": {
        "model_id": "jinaai/jina-clip-v2", 
        "collection_name": "jina_clip_v2_embedding", # کالکشن نسخه ۲
        "dimension": 1024,  # ابعاد نسخه ۲
        "type": "jina"
    }
}