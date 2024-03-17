import os
from pydantic import BaseModel
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class Pipeline(BaseModel):
    """
    Abstraction of different components of RAG pipeline
    """

    def create_document_nodes(self, type=None):
        # type = "sentencesplitter", "json"
        raise NotImplementedError
    
    def create_document_index(self, nodes: list[Nodes], store_directory=None):
        raise NotImplementedError
    


    def get_k_relevant_documents(self, document_index, top_k):
        raise NotImplementedError
    
    def get_n_reranked_documents(self, document_index, top_n):
        raise NotImplementedError
    
    def augment_context_with_prompt(self, context, prompt):
        raise NotImplementedError
    
    def generate_answer(self, prompt, query):
        raise NotImplementedError
    

    def load_embeddings(model_name, pooling):
        embed_model = HuggingFaceEmbedding(
                        model_name="BAAI/bge-m3",
                        pooling="mean" # "cls" is default
                    )
        return embed_model
    

    def load_index_from_disk(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        return index

    def get_k_relevant_documents_with_reranker(self, query, document_index, top_k, top_n, reranker_id="Dongjin-kr/ko-reranker"):
        reranker = FlagEmbeddingReranker(
            top_n=top_n,
            model=reranker_id,
        )

        raw_query_engine = document_index.as_query_engine(similarity_top_k=top_k, node_postprocessors=[reranker], verbose=True)

        relevent_docs = raw_query_engine.query(query)
        return relevent_docs        

    def get_context_from_relevant_documents(relevent_docs):
        context_list = []
        for r in relevent_docs.source_nodes:
            if r.score > 0:
                context_list.append(r.text)
        return context_list



## Uncomment this for new checkpoints
def get_prompt(question, raw_query_engine, prev_q=""):
    # prev_q is a must needed for some questions like this: What is the biggest cause of plaster revision? And please tell me how to solve this.”
    INSTRUCTION_PROMPT_TEMPLATE = """\
    ### System:
    벽지에 대한 고객 문의에 정확하고 유용한 답변을 작성한다. <질문>의 의도를 파악하여 정확하게 <보고서>만을 기반으로 답변하세요.

    ### User:
    <보고서>
    {CONTEXT}
    </보고서>
    지침사항을 반드시 지키고, <보고서>를 기반으로 <질문>에 답변하세요.
    <질문>
    {QUESTION}
    </질문>

    ### Assistant:
    """
    RESPONSE_TEMPLATE = """\
    {ANSWER}

    """

    response_1 = raw_query_engine.query(question)

    context_list = []
    for r in response_1.source_nodes:
      # print(r.score)
      if r.score > 0:
          if r.score <= 4 and len(context_list) >= 1:
              pass
          else:
              context_list.append(r.text)

    # Special case when the follow up question is junk
    if len(context_list) == 0:
        response_2 = raw_query_engine.query(prev_q + " " + question)
        for r in response_2.source_nodes:
            if r.score > 0:
                context_list.append(r.text)

    context = prev_q + "\n\n".join(context_list + [question])

    return INSTRUCTION_PROMPT_TEMPLATE.format(CONTEXT=context, QUESTION=question)
