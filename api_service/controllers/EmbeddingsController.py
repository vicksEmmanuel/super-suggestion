import time
import os
from pymilvus import connections, Collection
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from api_service.controllers.FileController import FileController
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

class EmbeddingsController:
    def __init__(self):
        connections.connect(
            host=os.getenv("MILVUS_HOST"),
            port=os.getenv("MILVUS_PORT")
        )
        self.collection_name = os.getenv("MILVUS_COLLECTION_NAME")
        self.collection = Collection(name=self.collection_name)
        self.embedding_dim = os.getenv("EMBEDDING_DIM")
        self.max_length = os.getenv("EMBEDDING_MAX_LENGTH")
        self.tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_PATH"))
        self.model = AutoModel.from_pretrained(os.getenv("MODEL_PATH"))

    def create_collection_in_milvus(self, dim=1024):
        if self.collection_name in utility.list_collections():
            print(f"Collection '{self.collection_name}' already exists.")
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name='key', dtype=DataType.STRING, is_primary=False, auto_id=False),
            ]
            schema = CollectionSchema(fields=fields, description="Embeddings")
            collection = Collection(name=self.collection_name, schema=schema)
            print(f"Collection '{self.collection_name}' created.")
        return Collection(name=self.collection_name)

    def create_index_for_collection(collection, field_name="embedding", index_type="IVF_FLAT", metric_type="L2", params={"nlist": 1024}):
        index_params = {"index_type": index_type, "metric_type": metric_type, "params": params}
        collection.create_index(field_name=field_name, index_params=index_params)
        print(f"Index created for field '{field_name}' in collection '{collection.name}'.")

    def wait_for_index_building_complete(collection_name):
        print("Waiting for index building to complete...")
        while True:
            status = utility.index_building_progress(collection_name)
            indexed_rows = status['indexed_rows']
            total_rows = status['total_rows']
            print(f"Index building progress: {indexed_rows}/{total_rows}")
            if indexed_rows >= total_rows:
                print("Index building complete.")
                break
            time.sleep(2)

    def download_workspace_and_create_embeddings(self, workspace_folder, key):
        # Download the workspace folder
        fileControls = FileController()
        workspace_path = fileControls.download_code_workspace_folder(workspace_folder, key)
        # Create embeddings
        self.create_embeddings_from_workspace(workspace_path, key)
        return {"response": "Success"}

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def generate_bert_embedding(self, text):
        input_text = f'{text}'
        batch_dict = self.tokenizer(input_text, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.squeeze().tolist()

    def create_embeddings_from_workspace(self, workspace_path, key):
        # Create collection
        self.create_collection_in_milvus(dim=self.embedding_dim)
        # Create embeddings
        embeddings = []
        keys = []
        for root, _, files in os.walk(workspace_path):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), "r") as f:
                        text = f.read()
                        chunks = self.chunk_text(text)
                        for i, chunk in enumerate(chunks):
                            embedding = self.generate_bert_embedding(chunk)
                            embeddings.append(embedding)
                            keys.append(f"{key}_{i}")
        self.insert_embeddings_to_milvus(embeddings, keys)
        return {"response": "Success"}

    def chunk_text(self, text, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.max_length - 2  # Accounting for special tokens
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            if len(current_chunk) + len(word.split()) <= chunk_size:
                current_chunk.append(word)
            else:
                chunk = " ".join(current_chunk)
                chunks.append(chunk)
                current_chunk = [word]

        if current_chunk:
            chunk = " ".join(current_chunk)
            chunks.append(chunk)

        return chunks

    def insert_embeddings_to_milvus(self, embeddings, keys):
        data = [
            {"embedding": embedding, "key": key} for embedding, key in zip(embeddings, keys)
        ]
        self.collection.insert(data)
        self.create_index_for_collection(self.collection)
        self.wait_for_index_building_complete(self.collection_name)