# core/db_manager.py
from pymilvus import MilvusClient, DataType
import config

class DBManager:
    def __init__(self):
        try:
            self.client = MilvusClient(uri=config.MILVUS_URI)
            self._init_collection()
        except Exception as e:
            print(f"❌ Connection Error: {e}")

    def _init_collection(self):
        # اگر کالکشن نبود، بساز
        if not self.client.has_collection(config.COLLECTION_NAME):
            schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=config.DIMENSION)
            schema.add_field("path", DataType.VARCHAR, max_length=1024) # آدرس عکس

            index_params = self.client.prepare_index_params()
            index_params.add_index("vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})

            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                schema=schema,
                index_params=index_params
            )

    def insert_image(self, vector, path):
        data = [{"vector": vector, "path": path}]
        res = self.client.insert(config.COLLECTION_NAME, data)
        return res

    def search(self, vector, top_k=5):
        res = self.client.search(
            collection_name=config.COLLECTION_NAME,
            data=[vector],
            limit=top_k,
            output_fields=["path"]
        )
        return res[0]
    
    def count(self):
        return self.client.query(config.COLLECTION_NAME, filter="", output_fields=["count(*)"])[0]
    

    def get_all_data(self, limit=10000):
            """
            دریافت تمام داده‌ها (ID, Vector, Path) برای بررسی تکراری‌ها
            """
            # نکته: اگر دیتای شما خیلی زیاد است باید از iterator استفاده کرد، 
            # اما برای زیر ۱۰ هزار عکس این روش سریع و خوب است.
            res = self.client.query(
                collection_name=config.COLLECTION_NAME,
                filter="id >= 0",  # ترفند برای گرفتن همه داده‌ها
                output_fields=["vector", "path"],
                limit=limit
            )
            return res

    def delete_by_ids(self, id_list):
        """
        حذف رکوردها از میلووس بر اساس لیست ID
        """
        if not id_list:
            return
        
        # تبدیل لیست به فرمت مناسب فیلتر
        # id in [123, 456, 789]
        filter_expr = f"id in {id_list}"
        self.client.delete(
            collection_name=config.COLLECTION_NAME,
            filter=filter_expr
        )


    def search(self, vector, top_k=5, filter_expr=None):
            """
            Search for similar vectors with an optional metadata filter.
            :param filter_expr: String expression for Milvus filtering (e.g., "path like '%vacation%'")
            """
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            res = self.client.search(
                collection_name=config.COLLECTION_NAME,
                data=[vector],
                limit=top_k,
                filter=filter_expr,  # 👈 این خط جدید است: اعمال فیلتر روی جستجو
                output_fields=["path"],
                search_params=search_params
            )
            # Milvus returns a list of results (one list per query vector)
            return res[0]