import streamlit as st
import os
import glob
from PIL import Image
import torch
from tqdm import tqdm
import config
# ایمپورت کلاس‌های اصلی که قبلاً ساختیم
from core.ai_engine import AIEngine
from core.db_manager import DBManager

# --- تنظیمات صفحه ---
st.set_page_config(
    page_title="Neural Search Dashboard", 
    page_icon="🧠", 
    layout="wide"
)

st.title("🧠 Neural Search Dashboard")
st.markdown("### Manage your embeddings & Bulk Indexing")

# --- مقداردهی اولیه کلاس‌ها ---
try:
    ai = AIEngine()
    db = DBManager()
except Exception as e:
    st.error(f"❌ System Error: {e}")
    st.stop()

# --- SIDEBAR: تنظیمات ---
with st.sidebar:
    st.header("⚙️ Batch Config")
    
    # 1. انتخاب مدل برای اینسرت (قابلیت جدید)
    model_options = list(config.MODELS_CONFIG.keys())
    selected_model = st.selectbox(
        "Select Target Model:", 
        model_options, 
        index=2 # پیش‌فرض روی Jina v2
    )
    
    # نمایش اطلاعات مدل انتخاب شده
    target_info = config.MODELS_CONFIG[selected_model]
    st.info(f"Target Collection:\n`{target_info['collection_name']}`\nDimension: `{target_info['dimension']}`")
    
    st.divider()
    
    # 2. تنظیمات بچ (سرعت)
    batch_size = st.slider("Batch Size (Speed vs VRAM)", 16, 128, 64)
    
    # 3. کپشن (هشدار سرعت)
    enable_caption = st.checkbox("Generate Captions (⚠️ Very Slow)", value=False, help="Turning this on will make indexing 50x slower!")

# --- MAIN AREA: رابط کاربری ---

# ورودی مسیر دیتاست
default_path = config.IMAGE_STORAGE_PATH
dataset_path = st.text_input("📁 Dataset Path (Folder containing images):", value=default_path)

# نمایش وضعیت فعلی کالکشن
if st.button("📊 Check Collection Status"):
    try:
        col_name = target_info['collection_name']
        if db.client.has_collection(col_name):
            # دریافت تعداد رکوردها
            res = db.client.query(collection_name=col_name, output_fields=["count(*)"])
            count = res[0]["count(*)"]
            st.success(f"✅ Collection `{col_name}` exists with **{count}** records.")
        else:
            st.warning(f"⚠️ Collection `{col_name}` does not exist yet (Will be created on insert).")
    except Exception as e:
        st.error(f"Connection Error: {e}")

st.divider()

# دکمه شروع عملیات سنگین
if st.button("🚀 Start Batch Indexing", type="primary"):
    
    # 1. بررسی مسیر
    if not os.path.exists(dataset_path):
        st.error(f"❌ Path `{dataset_path}` not found!")
        st.stop()

    # 2. پیدا کردن تمام عکس‌ها
    st.write("📂 Scanning for images...")
    image_files = []
    # جستجوی تمام فرمت‌های رایج
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
    
    if not image_files:
        st.warning("No images found in the specified folder.")
        st.stop()
        
    st.info(f"found **{len(image_files)}** images. Starting indexing process with **{selected_model}**...")

    # 3. اطمینان از وجود کالکشن (ساختن آن در صورت نبودن)
    db.ensure_collection(selected_model)

    # 4. لود کردن مدل هوش مصنوعی (فقط یکبار)
    with st.spinner(f"Loading {selected_model} model..."):
        # تابع load_embedding_model را از کلاس AI Engine صدا می‌زنیم
        # این تابع خروجی سه تایی برمی‌گرداند: (model, processor, model_type)
        model_data = ai.load_embedding_model(selected_model)

    # 5. حلقه اصلی پردازش (Batch Loop)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(image_files)
    processed_count = 0
    
    # تقسیم فایل‌ها به دسته‌های کوچک (Batch)
    for i in range(0, total_files, batch_size):
        batch_paths = image_files[i : i + batch_size]
        
        # الف) محاسبه امبدینگ برای این دسته
        # نکته: ما تابع get_embedding را طوری نوشته بودیم که تکی کار می‌کرد.
        # برای سرعت بیشتر در Batch، بهتر است مستقیم از مدل استفاده کنیم یا 
        # اگر می‌خواهید کد تمیز بماند، یک تابع get_batch_embedding به AI Engine اضافه کنید.
        # اما اینجا برای سادگی، تک تک محاسبه می‌کنیم (یا می‌توانید کد AI Engine را ارتقا دهید)
        
        vectors = []
        valid_paths_in_batch = []
        captions = []

        for path in batch_paths:
            try:
                # تولید بردار
                vec = ai.get_embedding(model_key=selected_model, image=path)
                
                # تولید کپشن (اگر فعال باشد)
                cap = ""
                if enable_caption:
                    cap = ai.generate_caption(path)
                
                vectors.append(vec)
                valid_paths_in_batch.append(path)
                captions.append(cap)
                
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        # ب) اینسرت دسته‌ای در Milvus
        if vectors:
            try:
                # آماده‌سازی فرمت داده برای Milvus
                data_to_insert = []
                for idx, v in enumerate(vectors):
                    data_to_insert.append({
                        "vector": v,
                        "path": valid_paths_in_batch[idx],
                        "caption": captions[idx]
                    })
                
                # درج در دیتابیس
                col_name = target_info['collection_name']
                db.client.insert(col_name, data_to_insert)
                
                processed_count += len(data_to_insert)
            except Exception as e:
                st.error(f"DB Insert Error: {e}")

        # بروزرسانی نوار پیشرفت
        progress = min((i + batch_size) / total_files, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"🚀 Indexed {processed_count} / {total_files} images...")

    st.balloons()
    st.success(f"🎉 Batch Indexing Completed! Successfully indexed {processed_count} images into `{target_info['collection_name']}`.")