# pages/2_🔍_Search.py
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper 
from core.ai_engine import AIEngine
from core.db_manager import DBManager
import os
import config

st.set_page_config(page_title="Search", page_icon="🔍", layout="wide")
st.title("🔍 Advanced Search Engine")

try:
    ai = AIEngine()
    db = DBManager()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Search Configuration")
    
    # انتخاب مدل (هر ۳ مدل)
    model_options = list(config.MODELS_CONFIG.keys())
    selected_model = st.selectbox(
        "Select Embedding Model:",
        model_options,
        index=2
    )
    st.info(f"Searching in: `{config.MODELS_CONFIG[selected_model]['collection_name']}`")
    
    st.markdown("---")
    
    search_type = st.radio(
        "Select Mode:", 
        ["Text Search 📝", "Image Search 🖼️", "Crop & Search ✂️", "Hybrid Search 🌪️"]
    )
    
    st.markdown("---")
    top_k = st.slider("Max Results:", 1, 50, 12)
    threshold = st.slider("Similarity Threshold:", 0.0, 1.0, 0.25, 0.01)

query_vector = None
milvus_filter = None 

# --- LOGIC ---
if search_type == "Text Search 📝":
    st.subheader(f"Semantic Text Search ({selected_model})")
    query = st.text_input("Describe the image:")
    if query:
        with st.spinner("Embedding text..."):
            query_vector = ai.get_embedding(model_key=selected_model, text=query)

elif search_type == "Image Search 🖼️":
    st.subheader(f"Image-to-Image Search ({selected_model})")
    up_img = st.file_uploader("Upload reference image:", type=['jpg', 'png', 'jpeg'])
    if up_img:
        image = Image.open(up_img).convert("RGB")
        st.image(image, width=250)
        if st.button("🔍 Search"):
            with st.spinner("Embedding image..."):
                query_vector = ai.get_embedding(model_key=selected_model, image=image)

elif search_type == "Crop & Search ✂️":
    st.subheader("Object Search (Crop)")
    up_img_crop = st.file_uploader("Upload image to crop:", type=['jpg', 'png', 'jpeg'], key="crop_upl")
    if up_img_crop:
        base_image = Image.open(up_img_crop).convert("RGB")
        col1, col2 = st.columns([2, 1])
        with col1:
            cropped_img = st_cropper(base_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
        with col2:
            st.image(cropped_img, caption="Target Object", use_container_width=True)
            if st.button("🔍 Search Object"):
                with st.spinner("Embedding object..."):
                    query_vector = ai.get_embedding(model_key=selected_model, image=cropped_img)

elif search_type == "Hybrid Search 🌪️":
    st.subheader("Hybrid Search (Vector + Metadata)")
    col_input, col_filter = st.columns(2)
    with col_input:
        st.markdown("### 1. Visual/Semantic Input")
        hybrid_mode = st.radio("Input Type:", ["Text", "Image"], horizontal=True)
        if hybrid_mode == "Text":
            h_text = st.text_input("Query:", key="h_text")
            if h_text:
                query_vector = ai.get_embedding(model_key=selected_model, text=h_text)
        else:
            h_img = st.file_uploader("Image:", type=['jpg', 'png'], key="h_img")
            if h_img:
                img_obj = Image.open(h_img).convert("RGB")
                st.image(img_obj, width=150)
                query_vector = ai.get_embedding(model_key=selected_model, image=img_obj)

    with col_filter:
        st.markdown("### 2. Hard Filter")
        filter_text = st.text_input("Filename pattern:")
        if filter_text:
            milvus_filter = f"path like '%{filter_text}%'"
            st.code(f"Filter: {milvus_filter}", language="sql")

# --- RESULTS ---
if query_vector is not None:
    st.divider()
    if search_type == "Hybrid Search 🌪️" and not st.button("🚀 Run Search"): st.stop()

    st.subheader("Results")
    with st.spinner(f"Searching in collection: {config.MODELS_CONFIG[selected_model]['collection_name']}..."):
        raw_results = db.search(
            model_key=selected_model, 
            vector=query_vector, 
            top_k=top_k, 
            filter_expr=milvus_filter
        )
    
    valid_results = [res for res in raw_results if res['distance'] >= threshold]
            
    if not valid_results:
        st.warning(f"🚫 No results found above threshold {threshold}.")
    else:
        st.success(f"Found {len(valid_results)} matches.")
        cols = st.columns(4)
        for i, res in enumerate(valid_results):
            score = res['distance']
            path = res['entity']['path']
            caption = res['entity'].get('caption', '')
            filename = os.path.basename(path)
            
            with cols[i % 4]:
                color = "green" if score > 0.6 else "orange"
                st.markdown(f"**{filename}**")
                st.caption(f":{color}[Score: {score:.4f}]")
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                    if caption: st.info(f"📄 {caption[:40]}...")
                else:
                    st.error("File missing")