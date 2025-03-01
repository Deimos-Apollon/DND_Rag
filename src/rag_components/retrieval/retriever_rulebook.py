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
    """A transformation component for cleaning text data in nodes.
    """
    
    def __call__(self, nodes, **kwargs):
        """Process and clean text content in nodes.
        
        Args:
            nodes (List[TextNode]): List of text nodes to process
            
        Returns:
            List[TextNode]: Nodes with cleaned text content
        """
        for node in nodes:
            text = node.text.replace("\n", " ")
            text = re.sub(r"(\w+)- (\w+)", r"\1\2", text)  # Merge word separated by -\n
            node.text = re.sub(r"\s+", " ", text).strip()
        return nodes


def get_index(
    path_to_data_dir: str = None,
    embed_model: HuggingFaceEmbeddings = None,
    use_faiss: bool = False,
    faiss_persist_dir: str = None
) -> VectorStoreIndex:
    """Create or load a vector store index from documents.
    
    Supports both in-memory and FAISS-based vector stores. When using FAISS,
    can persist and load the index from disk for reuse.

    Args:
        path_to_data_dir (str): Path to directory containing documents
        embed_model (HuggingFaceEmbeddings): Embedding model to use
        use_faiss (bool): Use FAISS for vector storage if True
        faiss_persist_dir (str): Directory for persisting FAISS index

    Returns:
        VectorStoreIndex: Configured vector store index

    Raises:
        ValueError: If no data directory provided for new index creation
    """
    embed_model = embed_model or Settings.embed_model

    if use_faiss:
        # 312 matches dimension size for cointegrated/rubert-tiny2 embeddings
        embedding_dimension = 312  
        
        if faiss_persist_dir and os.path.exists(
            os.path.join(faiss_persist_dir, "index_store.json")
        ):
            vector_store = FaissVectorStore.from_persist_dir(faiss_persist_dir)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=faiss_persist_dir
            )
            return load_index_from_storage(storage_context=storage_context)
            
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
    else:
        vector_store = None

    if not path_to_data_dir:
        raise ValueError("Data directory path required for index creation")

    # Configure PDF loader and document parser
    loader = PyMuPDFReader()
    documents = SimpleDirectoryReader(
        input_dir=path_to_data_dir,
        file_extractor={".pdf": loader}
    ).load_data()

    # Set up text processing pipeline
    text_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=128
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents=documents,
        transformations=[text_parser, TextCleaner()],
        embed_model=embed_model,
        storage_context=storage_context
    )

    if use_faiss and faiss_persist_dir:
        index.storage_context.persist(persist_dir=faiss_persist_dir)

    return index
