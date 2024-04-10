from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import Settings
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from .retriever_factory import RetrieverFactory


retriever_factory = RetrieverFactory()

@retriever_factory.register("naive")
class NaiveRetriever():
    
    def __init__(self, index: VectorStoreIndex, embed_model: BaseEmbedding, reranker: FlagEmbeddingReranker=None, query_engine: RetrieverQueryEngine=None):
        self.index = index
        self.embed_model = embed_model
        self.reranker = reranker
        self.query_engine = query_engine


    @classmethod
    def load_index_from_disk(cls, PERSIST_DIR, embed_model_name, pooling):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        embed_model = HuggingFaceEmbedding(
                        model_name=embed_model_name,
                        pooling=pooling # "cls" is default
                    )

        index = load_index_from_storage(storage_context, embed_model=embed_model, llm=None)
        return cls(index=index, embed_model=embed_model)


    def load_reranker(self, top_n, reranker_id):
        self.reranker = FlagEmbeddingReranker(
                    top_n=top_n,
                    model=reranker_id,
                )

    def load_retriever_query_engine(self, top_k):
        Settings.llm = None
        retriever_query_engine = self.index.as_retriever(similarity_top_k=top_k, node_postprocessors=[self.reranker], verbose=True)
        self.query_engine = retriever_query_engine
    

    def get_k_relevant_documents_with_reranker(self, query):
        relevent_docs = self.query_engine.query(query)
        return relevent_docs        

        
    def get_context_from_relevant_documents(self, relevent_docs):
        context_list = []
        for r in relevent_docs.source_nodes:
            if r.score > 0:
                context_list.append(r.text)
        return context_list
