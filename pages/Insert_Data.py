# pages/1_📥_Insert_Data.py
import streamlit as st
import os
from PIL import Image
from core.ai_engine import AIEngine
from core.db_manager import DBManager
import config  # ایمپورت کردن تنظیمات

st.title("📥 Add Images to Database")

ai = AIEngine()
db = DBManager()

tab1, tab2 = st.tabs(["Single Upload 📤", "Batch Folder 📂"])

# --- تب اول: آپلود تکی و ذخیره در مسیر خاص ---
with tab1:
    st.markdown("### Upload and Save to Storage")
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file and st.button("Save & Index Image"):
        # 1. ساختن مسیر نهایی فایل
        # مطمئن می‌شویم پوشه وجود دارد
        os.makedirs(config.IMAGE_STORAGE_PATH, exist_ok=True) 
        
        # آدرس کامل فایل نهایی
        save_path = os.path.join(config.IMAGE_STORAGE_PATH, uploaded_file.name)
        
        # 2. ذخیره کردن فایل فیزیکی روی دیسک
        try:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Image saved locally at: `{save_path}`")
            
            # نمایش تصویر جهت اطمینان
            image = Image.open(save_path).convert("RGB")
            st.image(image, width=250)

            # 3. ساخت امبدینگ و ارسال به Milvus
            with st.spinner("Generating Embedding & Indexing..."):
                # به جای فایل آپلودی، مسیر فایل ذخیره شده را می‌دهیم
                vector = ai.get_embedding(image=save_path) 
                
                # اینسرت در میلووس با آدرس دقیق روی سرور
                db.insert_image(vector, save_path)
                
                st.balloons()
                st.success("🎉 Successfully indexed in Milvus!")
                
        except Exception as e:
            st.error(f"❌ Error saving file: {e}")

# --- تب دوم: پردازش پوشه (بدون تغییر) ---
with tab2:
    st.markdown("### Index Existing Folder")
    folder_path = st.text_input("Enter folder path:", value=config.IMAGE_STORAGE_PATH)
    
    if st.button("Start Batch Indexing"):
        if os.path.exists(folder_path):
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not files:
                st.warning("No images found in this folder.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                count = 0
                for i, file_path in enumerate(files):
                    status_text.text(f"Processing: {os.path.basename(file_path)}")
                    try:
                        # چک می‌کنیم اگر عکس خراب بود رد شود
                        vec = ai.get_embedding(image=file_path)
                        db.insert_image(vec, file_path)
                        count += 1
                    except Exception as e:
                        print(f"Error skipping {file_path}: {e}")
                    
                    progress_bar.progress((i + 1) / len(files))
                
                st.success(f"✅ Finished! Indexed {count} images from folder.")
        else:
            st.error("❌ Folder path does not exist.")