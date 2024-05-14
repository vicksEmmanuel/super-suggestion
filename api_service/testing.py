
from api_service.controllers.EmbeddingsController import EmbeddingsController


emb = EmbeddingsController()

collection = emb.create_collection_in_milvus()

# Ensure index creation and wait for it to build
emb.create_index_for_collection(collection)
emb.wait_for_index_building_complete(collection.name)

# Download workspace and create embeddings
emb.download_workspace_and_create_embeddings(".", "super-suggestion")