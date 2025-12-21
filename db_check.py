# check_db.py
from pymilvus import MilvusClient
import config

# تنظیمات
# TARGET_COLLECTION = "jina_clip_embedding" # 👈 کالکشنی که می‌خواهیم چک کنیم

TARGET_COLLECTION = "jina_clip_v2_embedding" # 👈 کالکشنی که می‌خواهیم چک کنیم

 
def check_collection():
    print(f"🕵️ Connecting to Milvus at {config.MILVUS_URI}...")
    
    try:
        client = MilvusClient(uri=config.MILVUS_URI)
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    # 1. آیا کالکشن وجود دارد؟
    if not client.has_collection(TARGET_COLLECTION):
        print(f"❌ Collection '{TARGET_COLLECTION}' DOES NOT EXIST!")
        print("Existing collections:", client.list_collections())
        return

    print(f"✅ Collection '{TARGET_COLLECTION}' found!")

    # 2. دریافت تعداد کل داده‌ها
    # نکته: Milvus گاهی تعداد دقیق را با کمی تاخیر آپدیت می‌کند، اما query دقیق است.
    res_count = client.query(
        collection_name=TARGET_COLLECTION,
        filter="id >= 0",
        output_fields=["count(*)"]
    )
    count = res_count[0]['count(*)']
    print(f"📊 Total Entities (Rows): {count}")

    if count == 0:
        print("⚠️ The collection is empty.")
        return

    # 3. نمایش ۵ داده آخر (برای اطمینان از صحت محتوا)
    print("\n📝 Last 5 Entries:")
    res_data = client.query(
        collection_name=TARGET_COLLECTION,
        filter="id >= 0",
        output_fields=["id", "path", "caption"], # فقط فیلدهای قابل خواندن
        limit=5
    )

    for item in res_data:
        print(f" - ID: {item['id']}")
        print(f"   📂 Path: {item['path']}")
        print(f"   💬 Caption: {item.get('caption', 'No Caption')}")
        print("   -------------------------")

if __name__ == "__main__":
    check_collection()