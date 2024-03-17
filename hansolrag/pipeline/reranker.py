from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

reranker = FlagEmbeddingReranker(
    top_n=4,
    model="Dongjin-kr/ko-reranker",
)

raw_query_engine = index.as_query_engine(similarity_top_k=6, node_postprocessors=[reranker], verbose=True)

response_1 = raw_query_engine.query("면진장치가 뭐야?")
response_1
