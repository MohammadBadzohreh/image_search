# # app.py
# import streamlit as st
# import os
# import glob
# from PIL import Image
# import torch
# import config
# from core.ai_engine import AIEngine
# from core.db_manager import DBManager

# st.set_page_config(page_title="Neural Search Dashboard", page_icon="🧠", layout="wide")
# st.title("🧠 Neural Search Dashboard")

# try:
#     ai = AIEngine()
#     db = DBManager()
# except Exception as e:
#     st.error(f"❌ System Error: {e}")
#     st.stop()

# # --- SIDEBAR ---
# with st.sidebar:
#     st.header("⚙️ Batch Config")
    
#     # لیست مدل‌ها حالا شامل Llama-Nemo هم هست
#     model_options = list(config.MODELS_CONFIG.keys())
#     # پیش‌فرض روی آخرین مدل (احتمالا Nemo)
#     selected_model = st.selectbox("Select Target Model:", model_options, index=len(model_options)-1)
    
#     target_info = config.MODELS_CONFIG[selected_model]
#     st.info(f"Target Collection:\n`{target_info['collection_name']}`\nDim: `{target_info['dimension']}`")
    
#     st.divider()
#     # مدل Nemo سنگین است، بهتر است بچ سایز کوچکتر باشد
#     batch_size = st.slider("Batch Size", 4, 128, 32)
#     enable_caption = st.checkbox("Generate Captions (Slow)", value=False)

# # --- MAIN ---
# default_path = config.IMAGE_STORAGE_PATH
# dataset_path = st.text_input("📁 Dataset Path:", value=default_path)

# if st.button("📊 Check Status"):
#     try:
#         col_name = target_info['collection_name']
#         if db.client.has_collection(col_name):
#             count = db.client.query(col_name, output_fields=["count(*)"])[0]["count(*)"]
#             st.success(f"✅ `{col_name}` has **{count}** records.")
#         else:
#             st.warning(f"⚠️ `{col_name}` does not exist yet.")
#     except Exception as e:
#         st.error(f"Error: {e}")

# st.divider()

# if st.button("🚀 Start Batch Indexing", type="primary"):
#     if not os.path.exists(dataset_path):
#         st.error("Path not found!")
#         st.stop()

#     st.write("📂 Scanning images...")
#     image_files = []
#     for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
#         image_files.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
    
#     if not image_files:
#         st.warning("No images found.")
#         st.stop()
        
#     st.info(f"Found **{len(image_files)}** images. Indexing with **{selected_model}**...")
#     db.ensure_collection(selected_model)

#     with st.spinner(f"Loading {selected_model}..."):
#         # لود مدل
#         _ = ai.load_embedding_model(selected_model)

#     progress_bar = st.progress(0)
#     status_text = st.empty()
#     total_files = len(image_files)
#     processed_count = 0
    
#     for i in range(0, total_files, batch_size):
#         batch_paths = image_files[i : i + batch_size]
#         vectors = []
#         valid_paths = []
#         captions = []

#         for path in batch_paths:
#             try:
#                 # محاسبه امبدینگ (برای Nemo عکس مستقیم ارسال می‌شود)
#                 vec = ai.get_embedding(model_key=selected_model, image=path)
                
#                 cap = ""
#                 if enable_caption:
#                     cap = ai.generate_caption(path)
                
#                 if vec is not None:
#                     vectors.append(vec)
#                     valid_paths.append(path)
#                     captions.append(cap)
#             except Exception as e:
#                 # print(f"Error: {e}")
#                 continue
        
#         if vectors:
#             data = []
#             for idx, v in enumerate(vectors):
#                 data.append({"vector": v, "path": valid_paths[idx], "caption": captions[idx]})
            
#             try:
#                 db.client.insert(target_info['collection_name'], data)
#                 processed_count += len(data)
#             except Exception as e:
#                 st.error(f"DB Error: {e}")

#         progress = min((i + batch_size) / total_files, 1.0)
#         progress_bar.progress(progress)
#         status_text.text(f"🚀 Indexed {processed_count} / {total_files}...")

#     st.balloons()
#     st.success("Done!")


# app.py
import streamlit as st
import os
import glob
from PIL import Image
import torch
import config
from core.ai_engine import AIEngine
from core.db_manager import DBManager

# --- تنظیمات صفحه ---
st.set_page_config(page_title="Neural Search Dashboard", page_icon="🧠", layout="wide")
st.title("🧠 Neural Search Dashboard")

# --- مقداردهی اولیه موتورها ---
try:
    ai = AIEngine()
    db = DBManager()
except Exception as e:
    st.error(f"❌ System Error: {e}")
    st.stop()

# --- SIDEBAR: تنظیمات ---
with st.sidebar:
    st.header("⚙️ Batch Config")
    
    # 1. انتخاب مدل
    model_options = list(config.MODELS_CONFIG.keys())
    # پیش‌فرض روی آخرین مدل (Llama-Nemo)
    selected_model = st.selectbox("Select Target Model:", model_options, index=len(model_options)-1)
    
    # نمایش اطلاعات مدل
    target_info = config.MODELS_CONFIG[selected_model]
    st.info(f"Target Collection:\n`{target_info['collection_name']}`\nDim: `{target_info['dimension']}`")
    
    st.divider()
    
    # 2. تنظیمات بچ (Batch Size)
    # برای مدل‌های سنگین مثل Nemo، عدد کمتر (مثلاً 8 یا 16) بهتر است
    batch_size = st.slider("Batch Size", 4, 128, 16)
    
    # 3. تولید کپشن (اختیاری)
    enable_caption = st.checkbox("Generate Captions (Optional)", value=False)

# --- MAIN AREA ---
default_path = config.IMAGE_STORAGE_PATH
dataset_path = st.text_input("📁 Dataset Path:", value=default_path)

# دکمه بررسی وضعیت دیتابیس
if st.button("📊 Check Status"):
    try:
        col_name = target_info['collection_name']
        if db.client.has_collection(col_name):
            res = db.client.query(collection_name=col_name, output_fields=["count(*)"])
            count = res[0]["count(*)"]
            st.success(f"✅ Collection `{col_name}` exists with **{count}** records.")
        else:
            st.warning(f"⚠️ Collection `{col_name}` does not exist yet. It will be created automatically.")
    except Exception as e:
        st.error(f"Error checking DB: {e}")

st.divider()

# --- دکمه شروع عملیات ---
if st.button("🚀 Start Batch Indexing", type="primary"):
    # 1. بررسی مسیر فایل‌ها
    if not os.path.exists(dataset_path):
        st.error(f"❌ Path `{dataset_path}` not found!")
        st.stop()

    st.write("📂 Scanning for images...")
    image_files = []
    # جستجوی فرمت‌های مختلف
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
    
    if not image_files:
        st.warning("No images found in the specified folder.")
        st.stop()
        
    st.info(f"Found **{len(image_files)}** images. Starting indexing with **{selected_model}**...")
    
    # 2. اطمینان از وجود دیتابیس
    db.ensure_collection(selected_model)

    # 3. لود کردن مدل (زمان‌بر برای مدل‌های بزرگ)
    with st.spinner(f"Loading {selected_model}... (Please wait)"):
        try:
            ai.load_embedding_model(selected_model)
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.stop()

    # 4. شروع حلقه پردازش
    progress_bar = st.progress(0)
    status_text = st.empty()
    error_container = st.empty()
    
    total_files = len(image_files)
    processed_count = 0
    error_count = 0
    
    # حلقه اصلی (Batch Loop)
    for i in range(0, total_files, batch_size):
        batch_paths = image_files[i : i + batch_size]
        
        vectors = []
        valid_paths = []
        captions = []

        # پردازش هر عکس در بچ
        for path in batch_paths:
            try:
                # الف) تولید بردار (این تابع خودش لاجیک Nemo/CLIP را هندل می‌کند)
                vec = ai.get_embedding(model_key=selected_model, image=path)
                
                # ب) تولید کپشن (اگر تیک زده باشید)
                cap = ""
                if enable_caption:
                    cap = ai.generate_caption(path)
                
                if vec is not None:
                    vectors.append(vec)
                    valid_paths.append(path)
                    captions.append(cap)
            
            except Exception as e:
                # نمایش خطا در کنسول (ترمینال) برای دیباگ
                print(f"❌ Error processing {path}: {e}")
                error_count += 1
                if error_count <= 5: # فقط ۵ ارور اول را در صفحه نشان بده که شلوغ نشود
                    error_container.warning(f"Skipped {os.path.basename(path)}: {e}")
                continue
        
        # ج) ذخیره در دیتابیس (فقط اگر بردار معتبری ساخته شده باشد)
        if vectors:
            # ساخت ساختار داده مناسب برای Milvus
            data_to_insert = []
            for idx, v in enumerate(vectors):
                data_to_insert.append({
                    "vector": v,
                    "path": valid_paths[idx],
                    "caption": captions[idx]
                })
            
            try:
                # اینسرت مستقیم برای سرعت بیشتر
                db.client.insert(target_info['collection_name'], data_to_insert)
                processed_count += len(data_to_insert)
            except Exception as e:
                st.error(f"❌ DB Insert Error: {e}")
                # اگر دیتابیس قطع شده باشد، ادامه دادن بی‌فایده است
                st.stop()

        # د) آپدیت نوار پیشرفت
        progress = min((i + batch_size) / total_files, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"🚀 Indexed {processed_count} / {total_files} images... (Errors: {error_count})")

    st.balloons()
    st.success(f"🎉 Done! Successfully indexed **{processed_count}** images.")
    if error_count > 0:
        st.warning(f"⚠️ Skipped {error_count} images due to errors. Check terminal logs for details.")