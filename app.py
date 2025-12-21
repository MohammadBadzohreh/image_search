# app.py
import streamlit as st
import os
import glob
from PIL import Image
import torch
import config
from core.ai_engine import AIEngine
from core.db_manager import DBManager

st.set_page_config(page_title="Neural Search Dashboard", page_icon="🧠", layout="wide")
st.title("🧠 Neural Search Dashboard")

try:
    ai = AIEngine()
    db = DBManager()
except Exception as e:
    st.error(f"❌ System Error: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Batch Config")
    
    # لیست مدل‌ها حالا شامل CLIPA-v2 هم هست
    model_options = list(config.MODELS_CONFIG.keys())
    selected_model = st.selectbox("Select Target Model:", model_options, index=0)
    
    target_info = config.MODELS_CONFIG[selected_model]
    st.info(f"Target Collection:\n`{target_info['collection_name']}`\nDim: `{target_info['dimension']}`")
    
    st.divider()
    batch_size = st.slider("Batch Size", 16, 128, 64)
    enable_caption = st.checkbox("Generate Captions (Slow)", value=False)

# --- MAIN ---
default_path = config.IMAGE_STORAGE_PATH
dataset_path = st.text_input("📁 Dataset Path:", value=default_path)

if st.button("📊 Check Status"):
    try:
        col_name = target_info['collection_name']
        if db.client.has_collection(col_name):
            count = db.client.query(col_name, output_fields=["count(*)"])[0]["count(*)"]
            st.success(f"✅ `{col_name}` has **{count}** records.")
        else:
            st.warning(f"⚠️ `{col_name}` does not exist yet.")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

if st.button("🚀 Start Batch Indexing", type="primary"):
    if not os.path.exists(dataset_path):
        st.error("Path not found!")
        st.stop()

    st.write("📂 Scanning images...")
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
    
    if not image_files:
        st.warning("No images found.")
        st.stop()
        
    st.info(f"Found **{len(image_files)}** images. Indexing with **{selected_model}**...")
    db.ensure_collection(selected_model)

    with st.spinner(f"Loading {selected_model}..."):
        model_data = ai.load_embedding_model(selected_model)

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(image_files)
    processed_count = 0
    
    for i in range(0, total_files, batch_size):
        batch_paths = image_files[i : i + batch_size]
        vectors = []
        valid_paths = []
        captions = []

        for path in batch_paths:
            try:
                # اینجا تابع get_embedding بر اساس CLIPA، SigLIP یا Jina کار می‌کند
                vec = ai.get_embedding(model_key=selected_model, image=path)
                
                cap = ""
                if enable_caption:
                    cap = ai.generate_caption(path)
                
                vectors.append(vec)
                valid_paths.append(path)
                captions.append(cap)
            except:
                continue
        
        if vectors:
            data = []
            for idx, v in enumerate(vectors):
                data.append({"vector": v, "path": valid_paths[idx], "caption": captions[idx]})
            
            db.client.insert(target_info['collection_name'], data)
            processed_count += len(data)

        progress = min((i + batch_size) / total_files, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"🚀 Indexed {processed_count} / {total_files}...")

    st.balloons()
    st.success("Done!")