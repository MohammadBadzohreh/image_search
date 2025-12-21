# core/db_manager.py
from pymilvus import MilvusClient, DataType
import config

class DBManager:
    def __init__(self):
        try:
            self.client = MilvusClient(uri=config.MILVUS_URI)
        except Exception as e:
            print(f"❌ Connection Error: {e}")

    def ensure_collection(self, model_key):
        cfg = config.MODELS_CONFIG[model_key]
        col_name = cfg["collection_name"]
        dim = cfg["dimension"]

        if not self.client.has_collection(col_name):
            print(f"🆕 Creating collection '{col_name}' with dim={dim}...")
            
            schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field("path", DataType.VARCHAR, max_length=1024)
            schema.add_field("caption", DataType.VARCHAR, max_length=2048)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector", 
                index_type="HNSW", 
                metric_type="COSINE", 
                params={"M": 16, "efConstruction": 200}
            )

            self.client.create_collection(
                collection_name=col_name,
                schema=schema,
                index_params=index_params
            )
        return col_name

    def insert_image(self, model_key, vector, path, caption=""):
        col_name = self.ensure_collection(model_key)
        data = [{"vector": vector, "path": path, "caption": caption}]
        res = self.client.insert(col_name, data)
        return res

    def search(self, model_key, vector, top_k=5, filter_expr=None):
        cfg = config.MODELS_CONFIG[model_key]
        col_name = cfg["collection_name"]
        
        if not self.client.has_collection(col_name): return []

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        res = self.client.search(
            collection_name=col_name,
            data=[vector],
            limit=top_k,
            filter=filter_expr,
            output_fields=["path", "caption"],
            search_params=search_params
        )
        return res[0]

    def get_all_data(self, model_key, limit=10000):
        col_name = config.MODELS_CONFIG[model_key]["collection_name"]
        if not self.client.has_collection(col_name): return []
        
        res = self.client.query(
            collection_name=col_name,
            filter="id >= 0",
            output_fields=["vector", "path", "caption"],
            limit=limit
        )
        return res
    
    def delete_by_ids(self, model_key, id_list):
        if not id_list: return
        col_name = config.MODELS_CONFIG[model_key]["collection_name"]
        filter_expr = f"id in {id_list}"
        self.client.delete(collection_name=col_name, filter=filter_expr)