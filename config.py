# config.py
import torch

# --- تنظیمات عمومی ---
MILVUS_URI = "http://milvus-standalone:19530" 
IMAGE_STORAGE_PATH = "/home/jovyan/work/benchmark/data/flickr30k/Images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAPTION_MODEL = "Salesforce/blip-image-captioning-base" 

# --- تنظیمات مدل‌ها ---
MODELS_CONFIG = {
    # مدل‌های قبلی (SigLIP و Jina) سر جایشان هستند
    "SigLIP": {
        "model_id": "google/siglip-so400m-patch14-384",
        "collection_name": "siglip_gallery_v3_captioned",
        "dimension": 1152,
        "type": "siglip"
    },
    "Jina CLIP v1": {
        "model_id": "jinaai/jina-clip-v1", 
        "collection_name": "jina_clip_v1_embedding",
        "dimension": 768,
        "type": "jina"
    },
    "Jina CLIP v2": {
        "model_id": "jinaai/jina-clip-v2", 
        "collection_name": "jina_clip_v2_embedding",
        "dimension": 1024,
        "type": "jina"
    },
    # 👇 مدل جدید CLIPA-v2 (دقیقاً لینک شما)
    "CLIPA-v2 (ViT-H-14)": {
        # پیشوند hf-hub: یعنی مستقیم از لینک شما دانلود شود
        "model_id": "hf-hub:UCSC-VLAA/ViT-H-14-CLIPA-336-laion2B", 
        "collection_name": "clipa_v2_h14_336",
        "dimension": 1024, # ابعاد مدل ViT-H همیشه 1024 است
        "type": "open_clip" # 👈 نوع جدید برای پشتیبانی دقیق
    }
}