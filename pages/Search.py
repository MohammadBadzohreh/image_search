# import streamlit as st
# from PIL import Image
# from core.ai_engine import AIEngine
# from core.db_manager import DBManager
# import os

# st.set_page_config(page_title="Search", page_icon="🔍")
# st.title("🔍 Semantic Search")

# ai = AIEngine()
# db = DBManager()

# # --- تنظیمات سایدبار ---
# with st.sidebar:
#     st.header("Search Settings")
#     search_type = st.radio("Mode:", ["Text Search 📝", "Image Search 🖼️"])
#     top_k = st.slider("Max Results (Top-K):", 1, 20, 10)
    
#     # 👇 این بخش جدید است: تعیین حساسیت جستجو
#     st.markdown("---")
#     threshold = st.slider(
#         "Similarity Threshold (Hagh-e-ghabol):", 
#         min_value=0.0, 
#         max_value=1.0, 
#         value=0.25, 
#         step=0.01,
#         help="Higher value means stricter search. Lower value shows more results."
#     )

# query_vector = None
# results = []

# # --- دریافت ورودی ---
# if search_type == "Text Search 📝":
#     query = st.text_input("Describe what you are looking for:")
#     if query:
#         with st.spinner("Processing text..."):
#             query_vector = ai.get_embedding(text=query)

# elif search_type == "Image Search 🖼️":
#     up_img = st.file_uploader("Upload an image to find similar ones:", type=['jpg', 'png', 'jpeg'])
#     if up_img:
#         image = Image.open(up_img).convert("RGB")
#         st.image(image, caption="Query Image", width=200)
#         with st.spinner("Processing image..."):
#             query_vector = ai.get_embedding(image=image)

# # --- انجام جستجو و فیلتر کردن ---
# if query_vector is not None:
#     # جستجو در میلووس
#     raw_results = db.search(query_vector, top_k=top_k)
    
#     # 👇 فیلتر کردن نتایج بر اساس Threshold
#     valid_results = []
#     for res in raw_results:
#         score = res['distance'] # در Cosine Similarity این همان شباهت است
#         if score >= threshold:
#             valid_results.append(res)
    
#     st.divider()
    
#     if not valid_results:
#         st.warning(f"🚫 No results found with similarity above {threshold}. Try lowering the threshold.")
#     else:
#         st.success(f"✅ Found {len(valid_results)} matches!")
        
#         # نمایش نتایج
#         cols = st.columns(3)
#         for i, res in enumerate(valid_results):
#             score = res['distance']
#             path = res['entity']['path']
            
#             with cols[i % 3]:
#                 # نمایش اسکور با رنگ‌بندی
#                 if score > 0.6:
#                     score_color = "green"
#                 elif score > 0.4:
#                     score_color = "orange"
#                 else:
#                     score_color = "red"
                
#                 st.markdown(f"**Similarity:** :{score_color}[{score:.4f}]")
                
#                 if os.path.exists(path):
#                     st.image(path, use_container_width=True)
#                     st.caption(os.path.basename(path))
#                 else:
#                     st.error(f"File not found: {path}")







# pages/2_🔍_Search.py
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper 
from core.ai_engine import AIEngine
from core.db_manager import DBManager
import os

st.set_page_config(page_title="Search", page_icon="🔍", layout="wide")
st.title("🔍 Advanced Search Engine")

# Initialize Engines
try:
    ai = AIEngine()
    db = DBManager()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("Search Configuration")
    
    # 👇 اضافه شدن Hybrid Search به لیست
    search_type = st.radio(
        "Select Mode:", 
        [
            "Text Search 📝", 
            "Image Search 🖼️", 
            "Crop & Search ✂️", 
            "Hybrid Search 🌪️"
        ]
    )
    
    st.markdown("---")
    top_k = st.slider("Max Results:", 1, 50, 12)
    threshold = st.slider("Similarity Threshold:", 0.0, 1.0, 0.25, 0.01)

query_vector = None
milvus_filter = None  # متغیر نگهداری فیلتر

# =========================================
# MODE 1: TEXT SEARCH 📝
# =========================================
if search_type == "Text Search 📝":
    st.subheader("Semantic Text Search")
    query = st.text_input("Describe the image:")
    if query:
        with st.spinner("Embedding text..."):
            query_vector = ai.get_embedding(text=query)

# =========================================
# MODE 2: IMAGE SEARCH 🖼️
# =========================================
elif search_type == "Image Search 🖼️":
    st.subheader("Image-to-Image Search")
    up_img = st.file_uploader("Upload reference image:", type=['jpg', 'png', 'jpeg'])
    if up_img:
        image = Image.open(up_img).convert("RGB")
        st.image(image, width=250)
        if st.button("🔍 Search"):
            with st.spinner("Embedding image..."):
                query_vector = ai.get_embedding(image=image)

# =========================================
# MODE 3: CROP & SEARCH ✂️
# =========================================
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
                    query_vector = ai.get_embedding(image=cropped_img)

# =========================================
# MODE 4: HYBRID SEARCH 🌪️ (بخش جدید)
# =========================================
elif search_type == "Hybrid Search 🌪️":
    st.subheader("Hybrid Search (Vector + Metadata)")
    st.info("Combine visual similarity with filename filtering.")
    
    col_input, col_filter = st.columns(2)
    
    # ستون اول: ورودی برداری (متن یا عکس)
    with col_input:
        st.markdown("### 1. Visual/Semantic Input")
        hybrid_mode = st.radio("Input Type:", ["Text", "Image"], horizontal=True)
        
        if hybrid_mode == "Text":
            h_text = st.text_input("What does it look like?", key="h_text")
            if h_text:
                query_vector = ai.get_embedding(text=h_text)
        else:
            h_img = st.file_uploader("Upload Image:", type=['jpg', 'png'], key="h_img")
            if h_img:
                img_obj = Image.open(h_img).convert("RGB")
                st.image(img_obj, width=150)
                query_vector = ai.get_embedding(image=img_obj)

    # ستون دوم: فیلتر دقیق (Metadata)
    with col_filter:
        st.markdown("### 2. Hard Filter (Metadata)")
        st.markdown("Only show results where the filename contains:")
        filter_text = st.text_input("Filename pattern (e.g., 'vacation', '2024', '.png'):")
        
        if filter_text:
            # ساخت کوئری استاندارد Milvus
            # path like '%keyword%'
            milvus_filter = f"path like '%{filter_text}%'"
            st.code(f"Filter Expression:\n{milvus_filter}", language="sql")
        else:
            st.caption("No filter applied (Pure vector search).")

# =========================================
# COMMON RESULTS PROCESSOR
# =========================================
if query_vector is not None:
    st.divider()
    
    # دکمه نهایی برای حالت هیبرید (چون ورودی‌ها جداست)
    if search_type == "Hybrid Search 🌪️" and not st.button("🚀 Run Hybrid Search"):
        st.stop()

    st.subheader("Results")
    
    with st.spinner("Searching Milvus..."):
        # 👇 ارسال فیلتر به دیتابیس
        raw_results = db.search(query_vector, top_k=top_k, filter_expr=milvus_filter)
    
    # فیلتر آستانه شباهت (Threshold)
    valid_results = [res for res in raw_results if res['distance'] >= threshold]
            
    if not valid_results:
        msg = f"🚫 No results found above threshold {threshold}."
        if milvus_filter:
            msg += f"\n(Filter '{milvus_filter}' might be too strict)"
        st.warning(msg)
    else:
        st.success(f"Found {len(valid_results)} matches.")
        cols = st.columns(4)
        for i, res in enumerate(valid_results):
            score = res['distance']
            path = res['entity']['path']
            filename = os.path.basename(path)
            
            with cols[i % 4]:
                color = "green" if score > 0.6 else "orange"
                st.markdown(f"**{filename}**")
                st.caption(f":{color}[Score: {score:.4f}]")
                
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.error("File missing")