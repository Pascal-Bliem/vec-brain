import os
from dataclasses import dataclass
from enum import Enum

if os.environ.get("USER_AGENT") is None:
    os.environ["USER_AGENT"] = "CustomAgent/1.0"


class ChatModelProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class VectorStoreType(Enum):
    PINECONE = "pinecone"
    DOCARRAY = "docarray"


@dataclass
class AssistantConfig:
    """Configuration for the Assistant.

    Attributes:
        chat_model_provider (ChatModelProvider): The provider of the chat model.
        openai_api_key (str): API key for OpenAI.
        model (str): Model name to be used.
        temperature (float): Sampling temperature for the model.
        doc_splitter_chunk_size (int): Size of chunks for document splitting.
        doc_splitter_chunk_overlap (int): Overlap size for document splitting.
        embedding_model (str): Model name for embeddings.
        embedding_dimensions (int): Dimensions of the embedding vectors.
        vector_store_type (VectorStoreType): Type of vector store to use.
        n_retrieval_results (int): Number of retrieval results to return.
        retrieval_score_threshold (float): Score threshold for retrieval results.
    """

    chat_model_provider: ChatModelProvider = ChatModelProvider.OPENAI
    openai_api_key: str = os.environ.get("OPENAI_API_KEY")
    model: str = (
        "gpt-3.5-turbo-1106"
        if chat_model_provider == ChatModelProvider.OPENAI
        else "llama3"
    )
    temperature: float = 0.2
    doc_splitter_chunk_size: int = 1000
    doc_splitter_chunk_overlap: int = 20
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimensions: int = 1536  # default for text-embedding-ada-002
    vector_store_type: VectorStoreType = VectorStoreType.PINECONE
    n_retrieval_results: int = 5
    retrieval_score_threshold: float = 0.87


@dataclass
class DBConfig:
    """Configuration for the Database.

    Attributes:
        pinecone_api_key (str): API key for Pinecone.
        vector_index_name (str): Name of the vector index.
        vector_similarity_metric (str): Metric for vector similarity.
        cloud_provider (str): Cloud provider name.
        cloud_region (str): Cloud region name.
    """

    pinecone_api_key: str = os.environ.get("PINECONE_API_KEY")
    vector_index_name: str = "vec-brain-index"
    vector_similarity_metric: str = "cosine"
    cloud_provider: str = "aws"
    cloud_region: str = "us-east-1"  # the only region available for pinecone free tier


assistant_config = AssistantConfig()
db_config = DBConfig()
