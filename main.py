# main.py
import streamlit as st
from core.db_manager import DBManager

st.set_page_config(page_title="Milvus AI Search", page_icon="🧠", layout="wide")

st.title("🧠 AI Image Search Engine")
st.markdown("### Powered by Milvus & SigLIP")

try:
    db = DBManager()
    # گرفتن تعداد داده‌ها (روش تقریبی یا دقیق بسته به نسخه Milvus)
    st.success("✅ Connected to Milvus Standalone")
    st.info("Select a page from the sidebar to start!")
except Exception as e:
    st.error(f"❌ Could not connect to Database. Ensure Docker is running.\nError: {e}")