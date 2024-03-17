from pydantic import BaseModel
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.indices.vector_store.base import VectorStoreIndex
# from llama_index.core.indices.base import BaseIndex

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import Settings

from typing import Any

class Retriever():
    
    def __init__(self, index: VectorStoreIndex, embed_model: BaseEmbedding, reranker: FlagEmbeddingReranker=None):
        self.index = index
        self.embed_model = embed_model
        self.reranker = reranker

    @classmethod
    def load_index_from_disk(cls, PERSIST_DIR, embed_model_name="BAAI/bge-m3", pooling="mean"):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        embed_model = HuggingFaceEmbedding(
                        model_name=embed_model_name,
                        pooling=pooling # "cls" is default
                    )

        index = load_index_from_storage(storage_context, embed_model=embed_model, llm=None)
        return cls(index=index, embed_model=embed_model)

    def load_reranker(self, top_n=3, reranker_id="Dongjin-kr/ko-reranker"):
        self.reranker = FlagEmbeddingReranker(
                    top_n=top_n,
                    model=reranker_id,
                )

    def get_k_relevant_documents_with_reranker(self, query, top_k):

        Settings.llm = None
        raw_query_engine = self.index.as_query_engine(similarity_top_k=top_k, node_postprocessors=[self.reranker], verbose=True)

        relevent_docs = raw_query_engine.query(query)
        return relevent_docs        


    def get_context_from_relevant_documents(self, relevent_docs):
        context_list = []
        for r in relevent_docs.source_nodes:
            if r.score > 0:
                context_list.append(r.text)
        return context_list

        

