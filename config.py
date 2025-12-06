# config.py
import torch

# Milvus Config
MILVUS_URI = "http://milvus-standalone:19530" 
COLLECTION_NAME = "siglip_gallery_v2"
DIMENSION = 1152

# AI Model Config
MODEL_NAME = "google/siglip-so400m-patch14-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 👇 مسیر ذخیره تصاویر آپلودی (دقیقاً مسیری که خواستید)
IMAGE_STORAGE_PATH = "/home/jovyan/work/benchmark/data/flickr30k/Images"