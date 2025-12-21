# pages/3_🧹_Cleanup.py
import streamlit as st
import os
from core.db_manager import DBManager
import config  # 👈 اضافه شد برای دسترسی به لیست مدل‌ها

st.set_page_config(page_title="Cleanup", page_icon="🧹")
st.title("🧹 Database Hygiene")

db = DBManager()

# --- SIDEBAR: انتخاب دیتابیس ---
st.sidebar.header("Target Database")
# 👇 لیست مدل‌ها را از کانفیگ می‌خوانیم
model_options = list(config.MODELS_CONFIG.keys())
selected_model = st.sidebar.selectbox(
    "Select Model/Collection to Clean:",
    model_options,
    index=2  # پیش‌فرض (مثلاً Jina v2)
)

# نمایش نام کالکشن جهت اطمینان
target_collection = config.MODELS_CONFIG[selected_model]["collection_name"]
st.sidebar.warning(f"⚠️ You are cleaning:\n`{target_collection}`")
st.sidebar.markdown("---")

tab1, tab2 = st.tabs(["Deduplication (تکراری‌ها)", "Broken Links (فایل‌های گم‌شده)"])

# ==========================================
# TAB 1: حذف داده‌های تکراری
# ==========================================
with tab1:
    st.markdown(f"### 1. Remove Duplicate Vectors in **{selected_model}**")
    st.info("Logic: If paths are identical, only remove DB record. If files are different copies, remove file too.")

    if st.button("🔍 Scan for Duplicates"):
        with st.spinner(f"Scanning collection {target_collection}..."):
            # 👇 ارسال مدل انتخابی به تابع get_all_data
            all_data = db.get_all_data(model_key=selected_model, limit=16000)
            
            unique_vectors = {}  # {vector_tuple: {'id': id, 'path': path}}
            duplicates = []      # List of items to delete
            
            if not all_data:
                st.warning("Database is empty or connection failed.")
            else:
                for item in all_data:
                    # تبدیل لیست بردار به تاپل برای قابل هش شدن
                    vec_signature = tuple(item['vector'])
                    
                    if vec_signature in unique_vectors:
                        # تکراری پیدا شد!
                        original = unique_vectors[vec_signature]
                        
                        # بررسی می‌کنیم آیا فایل فیزیکی‌شان هم یکی است؟
                        is_same_file = (original['path'] == item['path'])
                        
                        duplicates.append({
                            'id': item['id'],
                            'path': item['path'],
                            'is_same_file': is_same_file,
                            'original_id': original['id']
                        })
                    else:
                        # اولین بار است می‌بینیم (اصلی)
                        unique_vectors[vec_signature] = {'id': item['id'], 'path': item['path']}

                st.session_state['duplicates'] = duplicates
                
                if not duplicates:
                    st.success("✨ No duplicates found.")
                else:
                    st.warning(f"⚠️ Found {len(duplicates)} duplicates.")

    # نمایش و حذف
    if 'duplicates' in st.session_state and st.session_state['duplicates']:
        dups = st.session_state['duplicates']
        
        with st.expander("Show Details"):
            for d in dups[:10]:
                action = "Database Only" if d['is_same_file'] else "Disk & Database"
                st.write(f"🗑️ ID: {d['id']} | Action: {action} | Path: {d['path']}")

        if st.button("🚀 Confirm Delete"):
            progress_bar = st.progress(0)
            deleted_ids = []
            files_removed = 0
            
            for i, item in enumerate(dups):
                # 1. حذف فایل فیزیکی (فقط اگر فایل‌ها جداگانه باشند)
                if not item['is_same_file']:
                    try:
                        if os.path.exists(item['path']):
                            os.remove(item['path'])
                            files_removed += 1
                    except Exception as e:
                        print(f"Error deleting file: {e}")
                
                # 2. همیشه حذف از دیتابیس
                deleted_ids.append(item['id'])
                progress_bar.progress((i + 1) / len(dups))
            
            # 👇 ارسال مدل انتخابی برای حذف
            db.delete_by_ids(model_key=selected_model, id_list=deleted_ids)
            
            st.success(f"Done! Removed {len(deleted_ids)} records from {target_collection} and {files_removed} files from disk.")
            del st.session_state['duplicates']

# ==========================================
# TAB 2: لینک‌های شکسته
# ==========================================
with tab2:
    st.markdown(f"### 2. Fix Broken Links in **{selected_model}**")
    st.markdown("Finds records in Milvus where the image file is missing from disk.")
    
    if st.button("🕵️ Scan for Missing Files"):
        with st.spinner("Checking file system..."):
            # 👇 ارسال مدل انتخابی
            all_data = db.get_all_data(model_key=selected_model, limit=16000)
            broken_links = []
            
            if not all_data:
                st.warning("Database is empty.")
            else:
                for item in all_data:
                    if not os.path.exists(item['path']):
                        broken_links.append(item)
                
                st.session_state['broken_links'] = broken_links
                
                if broken_links:
                    st.error(f"❌ Found {len(broken_links)} records with missing files.")
                else:
                    st.success("✅ All database records point to valid files.")

    if 'broken_links' in st.session_state and st.session_state['broken_links']:
        broken = st.session_state['broken_links']
        
        with st.expander("View Missing Files"):
            for b in broken:
                st.code(b['path'])
        
        if st.button("🧹 Clean Broken Records from DB"):
            ids_to_remove = [item['id'] for item in broken]
            # 👇 ارسال مدل انتخابی برای حذف
            db.delete_by_ids(model_key=selected_model, id_list=ids_to_remove)
            
            st.success(f"Removed {len(ids_to_remove)} broken records form {target_collection}.")
            del st.session_state['broken_links']