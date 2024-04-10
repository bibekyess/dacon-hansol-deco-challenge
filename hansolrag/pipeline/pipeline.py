from pydantic import BaseModel
from typing import Union, List
from hansolrag.retriever import retriever_factory
from hansolrag.generator import Generator

class Pipeline(BaseModel):
    """
    Abstraction of different components of RAG pipeline
    """

    def answer(self, question: Union[str, List],  config_path: str=None) -> List[str]:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        retriever = config.get("retriever", None)
        if retriever is None:
            raise ValueError("Please define retriever in the config")
        reranker = config.get("reranker", None)
        if reranker is None:
            raise ValueError("Please define reranker in the config")

        retriever_type = retriever['type']
        index_dir = retriever['index_dir']
        embed_model_name = retriever['model_name']
        pooling = retriever.get('pooling', 'cls')
        top_k = retriever['top_k']

        reranker_model_name = reranker['model_name']
        top_n = reranker['top_n']

        Retriever = retriever_factory.create_retriever(retriever_type)
        retriever = Retriever.load_index_from_disk(PERSIST_DIR=index_dir, embed_model_name=embed_model_name, pooling=pooling)
        retriever.load_reranker(top_n=top_n, reranker_id= reranker_model_name)
        retriever.load_retriever_query_engine(top_k=top_k)

        generator = Generator.from_config(config)

        if isinstance(question, str):
            question = [question]

        return generator.get_output(retriever, question)
