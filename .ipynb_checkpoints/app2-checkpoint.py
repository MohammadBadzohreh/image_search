import os
import glob
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, SiglipModel
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

# --- CONFIGURATION ---
# 1. Connect to your Docker container
# ⚠️ UPDATED: Using the internal Docker IP we found
MILVUS_URI = "http://milvus-standalone:19530"
COLLECTION_NAME = "siglib_gallery"

# 2. Path to your images
IMAGE_DATABASE_PATH = "/home/jovyan/work/benchmark/data/flickr30k/Images"

# 3. Model Config (SigLIP)
MODEL_NAME = "google/siglip-so400m-patch14-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MAIN SCRIPT ---
def main():
    print(f"🚀 Initializing on {DEVICE}...")

    # 1. Load AI Model
    print(f"📥 Loading model: {MODEL_NAME}...")
    model = SiglipModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # 2. Auto-Detect Dimension
    dummy_input = processor(text=["test"], return_tensors="pt", padding="max_length", max_length=10).to(DEVICE)
    with torch.no_grad():
        embedding_dim = model.get_text_features(**dummy_input).shape[-1]
    print(f"📏 Detected Embedding Dimension: {embedding_dim}")

    # 3. Connect to Milvus
    try:
        client = MilvusClient(uri=MILVUS_URI)
        print(f"✅ Connected to Milvus at {MILVUS_URI}!")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        print("💡 Hint: Check if the IP '192.168.96.4' is still correct by running: sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' milvus-standalone")
        return

    # 4. Create Collection (Robust Method)
    if client.has_collection(COLLECTION_NAME):
        print(f"⚠️ Collection '{COLLECTION_NAME}' already exists. Appending data...")
    else:
        print(f"🆕 Creating collection '{COLLECTION_NAME}'...")
        
        # We explicitly define schema to ensure 'path' is stored as a string
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
        schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=1024)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector", 
            index_type="HNSW", 
            metric_type="COSINE", 
            params={"M": 16, "efConstruction": 200}
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    # 5. Gather Images
    if not os.path.exists(IMAGE_DATABASE_PATH):
        print(f"❌ Error: Path '{IMAGE_DATABASE_PATH}' not found!")
        # Try to help debug path issues
        print(f"   Current working directory is: {os.getcwd()}")
        return

    image_paths = []
    # Added .JPG and .PNG case sensitivity check just in case
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DATABASE_PATH, "**", ext), recursive=True))
    
    if len(image_paths) == 0:
        print("❌ No images found. Check your IMAGE_DATABASE_PATH.")
        return

    print(f"📂 Found {len(image_paths)} images. Starting Indexing...")

    # 6. Insert Loop (Batch Processing)
    BATCH_SIZE = 64
    total_inserted = 0

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        images = []
        valid_paths = []

        # Load images
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Skipping bad image: {path}")
                continue

        if not images:
            continue

        # Generate Embeddings
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            # Normalize (Crucial for Cosine Similarity)
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        # Prepare Data for Milvus
        data = []
        feature_list = features.cpu().numpy()
        
        for idx, vector in enumerate(feature_list):
            data.append({
                "vector": vector,
                "path": valid_paths[idx]
            })

        # Insert into Database
        try:
            res = client.insert(collection_name=COLLECTION_NAME, data=data)
            total_inserted += len(data)
        except Exception as e:
            print(f"Error inserting batch: {e}")

    print(f"\n🎉 Success! Inserted {total_inserted} images into Milvus collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()