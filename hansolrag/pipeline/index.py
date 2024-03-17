from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)


PERSIST_DIR = "/home/bibekali/dacon-hansol-deco-challenge/hansolrag/index/train-vector-index-storage-chunk-size-1295"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)
