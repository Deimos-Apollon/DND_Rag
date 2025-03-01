import os
import re

import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings

from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, TransformComponent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.faiss import FaissVectorStore


Settings.embed_model = HuggingFaceEmbeddings(
    model_name="cointegrated/rubert-tiny2"
)


class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            text = node.text.replace('\n', ' ')
            # Объединяем разделённые переносами слова (например, "ко-\nнец" -> "конец")
            text = re.sub(r'(\w+)- (\w+)', r'\1\2', text)
            node.text = re.sub(r'\s+', ' ', text).strip()
        return nodes


def get_index(
    path_to_data_dir: str = None,
    embed_model=None,
    use_faiss: bool = False,
    faiss_persist_dir: str = None
):
    """
    Создает или загружает индекс.

    Параметры:
        path_to_data_dir (str): Путь к директории с документами. Если None, документы не загружаются.
        embed_model: Модель для создания эмбеддингов. Если None, используется модель из Settings.
        use_faiss (bool): Если True, используется FAISS для хранения векторов.
        faiss_persist_dir (str): Путь для сохранения/загрузки FAISS индекса. Если None, индекс не сохраняется.
    """
    if not embed_model:
        embed_model = Settings.embed_model

    if use_faiss:
        d = 312  # Размерность для rubert-tiny2
        if faiss_persist_dir and os.path.exists(os.path.join(faiss_persist_dir, "index_store.json")):
            vector_store = FaissVectorStore.from_persist_dir(faiss_persist_dir)
            
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=faiss_persist_dir
            )
            
            index = load_index_from_storage(storage_context=storage_context)
            return index
        else:
            faiss_index = faiss.IndexFlatL2(d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
    else:
        vector_store = None

    if not path_to_data_dir:
        raise ValueError("Не указан путь к данным для создания индекса.")

    loader = PyMuPDFReader()
    file_extractor = {".pdf": loader}
    documents = SimpleDirectoryReader(
        path_to_data_dir,
        file_extractor=file_extractor
    ).load_data()

    text_parser = SentenceSplitter(
        chunk_size=512, chunk_overlap=64
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_parser, TextCleaner()],
        embed_model=embed_model,
        storage_context=storage_context
    )

    if use_faiss and faiss_persist_dir:
        index.storage_context.persist(faiss_persist_dir)
    return index