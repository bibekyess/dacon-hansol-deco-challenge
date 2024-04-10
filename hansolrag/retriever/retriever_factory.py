class RetrieverFactory:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(retriever_class):
            self._registry[name] = retriever_class
            return retriever_class
        return decorator

    def create_retriever(self, name):
        retriever_class = self._registry.get(name)
        if retriever_class:
            return retriever_class
        else:
            raise ValueError(f"No retriever registered with name '{name}'")
