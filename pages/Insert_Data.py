# pages/1_📥_Insert_Data.py
import streamlit as st
import os
from PIL import Image
from core.ai_engine import AIEngine
from core.db_manager import DBManager
import config

st.set_page_config(page_title="Insert Data", page_icon="📥")
st.title("📥 Add Images & Captions")

ai = AIEngine()
db = DBManager()

# --- تنظیمات مدل ---
st.sidebar.header("Model Settings")

# خواندن لیست مدل‌ها از کانفیگ (هر ۳ مدل اینجا ظاهر می‌شوند)
model_options = list(config.MODELS_CONFIG.keys()) 
selected_model = st.sidebar.selectbox(
    "Select Embedding Model:",
    model_options,
    index=2  # پیش‌فرض روی گزینه آخر (Jina v2)
)

target_collection = config.MODELS_CONFIG[selected_model]["collection_name"]
st.sidebar.info(f"Target Collection:\n`{target_collection}`")

st.markdown(f"### Single Upload using **{selected_model}**")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    caption_mode = st.radio(
        "Caption Mode:", 
        ["No Caption 🚫", "Auto Caption 🤖", "Manual Caption ✍️"],
        horizontal=True
    )

    final_caption = ""
    if caption_mode == "Manual Caption ✍️":
        final_caption = st.text_area("Caption:", placeholder="Enter description...")

    if st.button("Save & Index"):
        os.makedirs(config.IMAGE_STORAGE_PATH, exist_ok=True)
        save_path = os.path.join(config.IMAGE_STORAGE_PATH, uploaded_file.name)
        
        try:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner(f"Processing with {selected_model}..."):
                # 1. کپشن
                if caption_mode == "Auto Caption 🤖":
                    st.info("🤖 AI is generating caption...")
                    final_caption = ai.generate_caption(save_path)
                    st.success(f"Generated Caption: **{final_caption}**")
                
                # 2. تولید بردار (با مدل انتخابی)
                vector = ai.get_embedding(model_key=selected_model, image=save_path)
                
                # 3. ذخیره در دیتابیس
                db.insert_image(model_key=selected_model, vector=vector, path=save_path, caption=final_caption)
                
                st.balloons()
                st.success(f"✅ Saved to `{target_collection}` successfully!")

        except Exception as e:
            st.error(f"Error: {e}")