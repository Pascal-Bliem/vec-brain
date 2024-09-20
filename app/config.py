import os
from dataclasses import dataclass

if os.environ.get("USER_AGENT") is None:
    os.environ["USER_AGENT"] = "CustomAgent/1.0"


@dataclass
class AssistantConfig:
    openai_api_key: str = os.environ.get("OPENAI_API_KEY")
    model: str = "gpt-3.5-turbo-1106"
    temperature: float = 0.2
    doc_splitter_chunk_size: int = 500
    doc_splitter_chunk_overlap: int = 10
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimensions: int = 1536
    vec_similarity_metric: str = "cosine_sim"
    n_retrieval_results: int = 5
    retrieval_score_threshold: float = 0.5


config = AssistantConfig()
