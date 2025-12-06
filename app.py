import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel
import os
import glob

# --- CONFIGURATION ---
# The path you provided
FLICKR_DATABASE_PATH = "/home/jovyan/work/benchmark/data/flickr30k/Images"

MODEL_NAME = "google/siglip-so400m-patch14-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- LOAD MODEL (Cached) ---
@st.cache_resource
def load_model():
    print(f"🚀 Loading {MODEL_NAME} on {DEVICE}...")
    model = SiglipModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor

model, processor = load_model()

# --- HELPER FUNCTIONS ---
def get_image_files(directory):
    """Finds all images in the Flickr directory."""
    # Look for common image formats
    extensions = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for ext in extensions:
        # Recursive search inside the path
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return files

def compute_embeddings(image_paths):
    """Encodes the database images into vectors."""
    all_embeddings = []
    valid_paths = []
    batch_size = 64  # Process 64 images at a time to be fast

    # Progress bar UI
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        current_batch_paths = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                current_batch_paths.append(path)
            except Exception as e:
                continue # Skip broken images

        if len(images) == 0:
            continue

        # Convert images to tensors
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            # Get the visual features
            features = model.get_image_features(**inputs)
            # Normalize them (Crucial for Cosine Similarity)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            all_embeddings.append(features.cpu())
            valid_paths.extend(current_batch_paths)

        # Update progress bar
        progress = min((i + batch_size) / len(image_paths), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Indexed {len(valid_paths)} / {len(image_paths)} images...")

    if all_embeddings:
        return torch.cat(all_embeddings), valid_paths
    return None, []

# --- MAIN APP LAYOUT ---
st.title("🔎 Flickr30k Neural Search")
st.write(f"**Database Path:** `{FLICKR_DATABASE_PATH}`")

# --- STEP 1: INDEXING ---
if 'db_vectors' not in st.session_state:
    st.info("⚠️ Database not loaded. Click the button below to index your Flickr images.")
    
    if st.button("🚀 Load & Index Database"):
        if not os.path.exists(FLICKR_DATABASE_PATH):
            st.error(f"❌ Error: The path `{FLICKR_DATABASE_PATH}` does not exist!")
            st.stop()
            
        files = get_image_files(FLICKR_DATABASE_PATH)
        st.write(f"Found {len(files)} images. Generating embeddings...")
        
        vectors, paths = compute_embeddings(files)
        
        if vectors is not None:
            # Store in session state (RAM)
            st.session_state['db_vectors'] = vectors.to(DEVICE)
            st.session_state['db_paths'] = paths
            st.success(f"✅ Successfully indexed {len(paths)} images!")
            st.rerun() # Refresh page to show search tools
        else:
            st.error("No valid images found to index.")

# --- STEP 2: SEARCH INTERFACE ---
else:
    st.success(f"✅ Database Ready ({len(st.session_state['db_paths'])} images indexed)")
    
    # Toggle between modes
    mode = st.radio("Select Search Mode:", ["📝 Text-to-Image", "🖼️ Image-to-Image"], horizontal=True)
    
    query_vector = None

    # --- MODE A: TEXT SEARCH ---
    if mode == "📝 Text-to-Image":
        text_input = st.text_input("Describe the image you want:", placeholder="e.g. A group of people dancing in the street")
        
        if text_input:
            with torch.no_grad():
                # Encode text
                inputs = processor(text=[text_input], return_tensors="pt").to(DEVICE)
                features = model.get_text_features(**inputs)
                query_vector = features / features.norm(p=2, dim=-1, keepdim=True)

    # --- MODE B: IMAGE SEARCH ---
    elif mode == "🖼️ Image-to-Image":
        uploaded = st.file_uploader("Upload an image to find similar ones", type=["jpg", "png", "jpeg"])
        
        if uploaded:
            in_image = Image.open(uploaded).convert("RGB")
            st.image(in_image, caption="Your Query Image", width=250)
            
            with torch.no_grad():
                # Encode image
                inputs = processor(images=in_image, return_tensors="pt").to(DEVICE)
                features = model.get_image_features(**inputs)
                query_vector = features / features.norm(p=2, dim=-1, keepdim=True)

    # --- STEP 3: PERFORM SEARCH ---
    if query_vector is not None:
        db_vectors = st.session_state['db_vectors']
        
        # Math: Dot product between query and all database vectors
        # Result is a list of similarity scores (higher is better)
        similarities = torch.matmul(query_vector, db_vectors.T).squeeze()
        
        # Get top 5 matches
        top_k = 5
        scores, indices = torch.topk(similarities, top_k)
        
        st.divider()
        st.subheader("🎯 Top Matches")
        
        # Display results in columns
        cols = st.columns(top_k)
        for i, col in enumerate(cols):
            idx = indices[i].item()
            score = scores[i].item()
            filepath = st.session_state['db_paths'][idx]
            filename = os.path.basename(filepath)
            
            with col:
                st.image(filepath, use_container_width=True)
                st.caption(f"**{score:.2%} Match**\n`{filename}`")