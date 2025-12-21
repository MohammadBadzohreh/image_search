# config.py
import torch

# Milvus Config
MILVUS_URI = "http://milvus-standalone:19530" 
# 👇 نام کالکشن را عوض کردیم تا فیلد جدید Caption ساخته شود
COLLECTION_NAME = "siglip_gallery_v3_captioned"
DIMENSION = 1152 
IMAGE_STORAGE_PATH = "/home/jovyan/work/benchmark/data/flickr30k/Images"

# AI Models Config
EMBEDDING_MODEL = "google/siglip-so400m-patch14-384"
# 👇 مدل جدید برای تولید متن
CAPTION_MODEL = "Salesforce/blip-image-captioning-base" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"