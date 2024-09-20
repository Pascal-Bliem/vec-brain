import re
from typing import Dict, List

import prompts
from config import AssistantConfig, config
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.docarray.in_memory import \
    DocArrayInMemorySearch
from langchain_core.documents.base import Document
from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                                    PromptTemplate)
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Assistant:
    def __init__(self, config: AssistantConfig = config):
        self.chat = ChatOpenAI(
            openai_api_key=config.openai_api_key,
            model=config.model,
            temperature=config.temperature,
            verbose=True,
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.system),
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

        self.vectorstore = DocArrayInMemorySearch.from_documents(
            documents=[],
            embedding=self.embeddings,
        )

        self.retriever = self.vectorstore.as_retriever(
            k=config.n_retrieval_results,
            score_threshold=config.retrieval_score_threshold,
        )

        document_prompt = PromptTemplate(
            input_variables=["page_content", "source", "title"],
            template=prompts.document_prompt,
        )
        document_chain = create_stuff_documents_chain(
            self.chat, self.prompt_template, document_prompt=document_prompt
        )

        retrieval_chain = RunnablePassthrough.assign(
            context=self._parse_retriever_input | self.retriever,
        ).assign(answer=document_chain)

        self.chain = retrieval_chain

    def _parse_retriever_input(self, params: Dict) -> str:
        if "messages" not in params or len(params["messages"]) == 0:
            return ""
        return params["messages"][-1].content

    def add_user_message_with_documents(self, message: str):
        self.chat_history.add_user_message(message)
        self._add_documents_from_user_message_to_store(message)

    def _add_documents_from_user_message_to_store(self, message: str):
        documents = []

        # load docs from URL
        if "http" in message:
            for url in re.findall(r"(https?://\S+)", message):
                documents += self._load_from_url(url)
            self.vectorstore.add_documents(documents)

    def _load_from_url(self, url: str) -> List[Document]:
        loader = WebBaseLoader(url)
        data = loader.load()

        return self.text_splitter.split_documents(data)
