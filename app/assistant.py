import re
from typing import Dict, List

import prompts
from config import AssistantConfig, VectorStoreType
from config import assistant_config as config

if config.vector_store_type == VectorStoreType.PINECONE:
    from langchain_pinecone import PineconeVectorStore
    from db import vector_index

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.docarray.in_memory import \
    DocArrayInMemorySearch
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                                    PromptTemplate)
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Assistant:
    """Assistant class to handle chat interactions and document retrieval."""

    def __init__(self, config: AssistantConfig = config):
        """
        Initializes the Assistant with the given configuration.

        Args:
            config (AssistantConfig): Configuration for the Assistant.
        """
        self.chat = ChatOpenAI(
            openai_api_key=config.openai_api_key,
            model=config.model,
            temperature=config.temperature,
            verbose=True,
        )

        self.prompt_template_rag = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.system_rag),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        self.prompt_template_no_rag = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.system_no_rag),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        query_transform_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                ("user", prompts.query_transform_prompt),
            ]
        )

        self.chat_history = ChatMessageHistory()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.doc_splitter_chunk_size,
            chunk_overlap=config.doc_splitter_chunk_overlap,
        )

        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config.openai_api_key,
            model=config.embedding_model,
        )

        if config.vector_store_type == VectorStoreType.PINECONE:
            self.vectorstore = PineconeVectorStore(
                index=vector_index, embedding=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": config.n_retrieval_results,
                    "score_threshold": config.retrieval_score_threshold,
                },
            )
        elif config.vector_store_type == VectorStoreType.DOCARRAY:
            self.vectorstore = DocArrayInMemorySearch.from_documents(
                documents=[],
                embedding=self.embeddings,
            )
            self.retriever = self.vectorstore.as_retriever(
                k=config.n_retrieval_results,
                score_threshold=config.retrieval_score_threshold,
            )
        else:
            raise NotImplementedError(
                f"Vector store type {config.vector_store_type} not implemented"
            )

        document_prompt = PromptTemplate(
            input_variables=["page_content", "source", "title"],
            template=prompts.document_prompt,
        )

        document_chain = create_stuff_documents_chain(
            self.chat, self.prompt_template_rag, document_prompt=document_prompt
        )

        retrieval_chain = RunnablePassthrough.assign(
            context=self._parse_retriever_input | self.retriever,
        ).assign(
            answer=RunnableBranch(
                (lambda params: params.get("context"), document_chain),
                self.prompt_template_no_rag | self.chat | StrOutputParser(),
            )
        )

        self.chain = retrieval_chain

    def _parse_retriever_input(self, params: Dict) -> str:
        """
        Parses the retriever input to extract the last message content.

        Args:
            params (Dict): Parameters containing chat messages.

        Returns:
            str: The content of the last message.
        """
        if "messages" not in params or len(params["messages"]) == 0:
            return ""
        return params["messages"][-1].content

    def add_user_message_with_documents(self, message: str):
        """
        Adds a user message to the chat history and processes any documents
        referenced in the message.

        Args:
            message (str): The user message.
        """
        self.chat_history.add_user_message(message)
        self._add_documents_from_user_message_to_store(message)

    def _add_documents_from_user_message_to_store(self, message: str):
        """
        Adds documents (anything the user may want to save in knowledge)
        store) from the user message to the vector store.
        Currently supports loading documents from URLs.

        Args:
            message (str): The user message containing document URLs.
        """
        documents = []

        # load docs from URL
        if "http" in message:
            for url in re.findall(r"(https?://\S+)", message):
                documents += self._load_from_url(url)
            self.vectorstore.add_documents(documents)

    def _load_from_url(self, url: str) -> List[Document]:
        """
        Loads content from URL and splits it into documents.

        Args:
            url (str): The URL to load content from.

        Returns:
            List[Document]: A list of loaded documents.
        """
        loader = WebBaseLoader(url)
        data = loader.load()

        return self.text_splitter.split_documents(data)
