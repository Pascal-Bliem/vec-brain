import time

from config import assistant_config, db_config
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=db_config.pinecone_api_key)

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if db_config.vector_index_name not in existing_indexes:
    pc.create_index(
        name=db_config.vector_index_name,
        dimension=assistant_config.embedding_dimensions,
        metric=db_config.vector_similarity_metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(db_config.vector_index_name).status["ready"]:
        time.sleep(1)

vector_index = pc.Index(db_config.vector_index_name)
