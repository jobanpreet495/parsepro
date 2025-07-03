from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.vectorstores.base import VectorStore
from pathlib import Path
from typing import List, Optional, Union, Type, Any, Dict
from langchain.schema import Document
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class DocumentIndexer:
    """
    Professional document indexer for RAG applications.
    
    Handles embedding generation and vector storage with support for
    multiple embedding models and vector databases.
    
    Attributes:
        embedding_model: The embedding model instance
        vector_store_class: The vector store class to use
        vector_store: The initialized vector store instance
    """
    
    # # Default configurations for different vector stores
    VECTOR_STORE_CONFIGS = {
        Chroma: {
            'persist_directory': './chroma_db',
            'collection_name': 'rag_documents'
        }
    }
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        vector_store_class: Type[VectorStore] = FAISS,
        vector_store_config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        **embedding_kwargs
    ):
        """
        Initialize the DocumentIndexer.
        
        Args:
            embedding_model: Embedding model instance (default: OpenAIEmbeddings)
            vector_store_class: Vector store class (default: FAISS)
            vector_store_config: Configuration for vector store
            api_key: API key for embedding service (e.g., OpenAI API key)
            **embedding_kwargs: Additional parameters for embedding model
        """
        # Initialize embedding model
        if embedding_model is None:
            # Set API key if provided
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            
            self.embedding_model = OpenAIEmbeddings(**embedding_kwargs)
        else:
            self.embedding_model = embedding_model
        
        self.vector_store_class = vector_store_class
        self.vector_store_config = vector_store_config or self.VECTOR_STORE_CONFIGS.get(
            vector_store_class, {}
        )
        self.vector_store = None
        
        logger.info(f"DocumentIndexer initialized with {self.embedding_model.__class__.__name__}")
        logger.info(f"Vector store: {vector_store_class.__name__}")
        logger.info(f"Vector store config: {self.vector_store_config}")
    
    def create_index(
        self, 
        documents: List[Document],
        index_name: Optional[str] = None,
        **vector_store_kwargs
    ) -> VectorStore:
        """
        Create vector index from documents.
        
        Args:
            documents: List of Document objects to index
            index_name: Name for the index (used by some vector stores)
            **vector_store_kwargs: Additional parameters for vector store
            
        Returns:
            Initialized vector store instance
            
        Raises:
            ValueError: If no documents provided
        """
        if not documents:
            raise ValueError("No documents provided for indexing")
        
        logger.info(f"Creating index from {len(documents)} documents...")
        
        try:
            # # Prepare vector store configuration
            config = self.vector_store_config.copy()
            config.update(vector_store_kwargs)
            
            # Handle different vector store types
            if self.vector_store_class == FAISS:
                self.vector_store = FAISS.from_documents(
                    documents,
                    self.embedding_model,
                   
                )
            elif self.vector_store_class == Chroma:
                if index_name:
                    config['collection_name'] = index_name
                self.vector_store = Chroma.from_documents(
                    documents,
                    self.embedding_model,
                    **config
                )
            else:
                # Generic vector store initialization
                self.vector_store = self.vector_store_class.from_documents(
                    documents,
                    self.embedding_model,
                    **config
                )
            
            logger.info(f"Successfully created index with {len(documents)} documents")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise
    
    def add_documents(
        self, 
        documents: List[Document],
       
    ) -> List[str]:
        """
        Add documents to existing index.
        
        Args:
            documents: List of Document objects to add
            
            
        Returns:
            List of document IDs
            
        Raises:
            ValueError: If no vector store exists or no documents provided
        """
        if self.vector_store is None:
            raise ValueError("No vector store exists. Create index first.")
        
        if not documents:
            raise ValueError("No documents provided for adding")
        
        logger.info(f"Adding {len(documents)} documents to existing index...")
        
        try:
            doc_ids = self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} documents")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def save_index(self, path: Union[str, Path]) -> None:
        """
        Save the vector index to disk.
        
        Args:
            path: Path to save the index
            
        Raises:
            ValueError: If no vector store exists
        """
        if self.vector_store is None:
            raise ValueError("No vector store exists to save")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(self.vector_store, FAISS):
                self.vector_store.save_local(str(path))
            elif isinstance(self.vector_store, Chroma):
                # Chroma persists automatically if persist_directory is set
                logger.info("Chroma database persisted automatically")
                return
            else:
                # Generic save method
                if hasattr(self.vector_store, 'save'):
                    self.vector_store.save(str(path))
                else:
                    logger.warning(f"Save method not available for {type(self.vector_store)}")
                    return
            
            logger.info(f"Index saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(
        self, 
        path: Union[str, Path],
        **kwargs
    ) -> VectorStore:
        """
        Load vector index from disk.
        
        Args:
            path: Path to load the index from
            **kwargs: Additional parameters for loading
            
        Returns:
            Loaded vector store instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Index path not found: {path}")
        
        try:
            if self.vector_store_class == FAISS:
                self.vector_store = FAISS.load_local(
                    str(path),
                    self.embedding_model,
                    allow_dangerous_deserialization = True,
                    **kwargs
                )
            elif self.vector_store_class == Chroma:
                config = self.vector_store_config.copy()
                config['persist_directory'] = str(path)
                config.update(kwargs)
                self.vector_store = Chroma(
                    embedding_function=self.embedding_model,
                    **config
                )
            else:
                # Generic load method
                if hasattr(self.vector_store_class, 'load'):
                    self.vector_store = self.vector_store_class.load(
                        str(path),
                        self.embedding_model,
                        **kwargs
                    )
                else:
                    raise NotImplementedError(f"Load method not implemented for {self.vector_store_class}")
            
            logger.info(f"Index loaded from {path}")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.vector_store is None:
            return {"index_exists": False}
        
        info = {
            "index_exists": True,
            "vector_store_type": type(self.vector_store).__name__,
            "embedding_model": type(self.embedding_model).__name__
        }
        
        # Try to get vector store specific info
        try:
            if isinstance(self.vector_store, FAISS):
                info["index_size"] = self.vector_store.index.ntotal
                info["vector_dimension"] = self.vector_store.index.d
            elif isinstance(self.vector_store, Chroma):
                collection = self.vector_store._collection
                info["collection_name"] = collection.name
                info["document_count"] = collection.count()
        except Exception as e:
            logger.warning(f"Could not retrieve detailed index info: {e}")
        
        return info