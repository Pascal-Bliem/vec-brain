import os
from dataclasses import dataclass
from enum import Enum

if os.environ.get("USER_AGENT") is None:
    os.environ["USER_AGENT"] = "CustomAgent/1.0"


class VectorStoreType(Enum):
    PINECONE = "pinecone"
    DOCARRAY = "docarray"


@dataclass
class AssistantConfig:
    openai_api_key: str = os.environ.get("OPENAI_API_KEY")
    model: str = "gpt-3.5-turbo-1106"
    temperature: float = 0.2
    doc_splitter_chunk_size: int = 1000
    doc_splitter_chunk_overlap: int = 20
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimensions: int = 1536  # default for text-embedding-ada-002
    vector_store_type: VectorStoreType = VectorStoreType.PINECONE
    n_retrieval_results: int = 5
    retrieval_score_threshold: float = 0.8


@dataclass
class DBConfig:
    pinecone_api_key: str = os.environ.get("PINECONE_API_KEY")
    vector_index_name: str = "vec-brain-index"
    vector_similarity_metric: str = "cosine"
    cloud_provider: str = "aws"
    cloud_region: str = "us-east-1"  # the only region available for pinecone free tier


assistant_config = AssistantConfig()
db_config = DBConfig()
