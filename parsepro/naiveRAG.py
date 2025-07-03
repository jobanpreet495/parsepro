from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Optional, Union, Any, Dict , Tuple , Type
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from parsepro.chunking import DocumentProcessor , TextSplitter
from parsepro.indexing import DocumentIndexer
from parsepro.retrieval import  DocumentRetriever
from  pathlib import Path
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class NaiveRAG:
    """
    Complete RAG document processing pipeline.
    
    Combines document loading, text splitting, indexing, and retrieval
    into a unified pipeline for complete RAG implementation.
    
    Attributes:
        document_processor: DocumentProcessor instance
        text_splitter: TextSplitter instance
        indexer: DocumentIndexer instance
        retriever: DocumentRetriever instance
    """
    
    def __init__(
        self,
        document_loader: Type = PyPDFLoader,
        text_splitter: Type = RecursiveCharacterTextSplitter,
        embedding_model: Optional[Any] = None,
        vector_store_class: Type[VectorStore] = FAISS,
        vector_store_config: Optional[Dict[str, Any]] = None,
        retrieval_params: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        **splitter_kwargs
    ):
        """
        Initialize the complete RAG document pipeline.
        
        Args:
            document_loader: Document loader class
            text_splitter: Text splitter class
            embedding_model: Embedding model instance
            vector_store_class: Vector store class
            vector_store_config: Configuration for vector store
            retrieval_params: Default parameters for retrieval
            api_key: API key for embedding service (e.g., OpenAI API key)
            **splitter_kwargs: Default parameters for text splitter
        """
        # Initialize components
        self.document_processor = DocumentProcessor(document_loader)
        self.text_splitter = TextSplitter(text_splitter, **splitter_kwargs)
        self.indexer = DocumentIndexer(
            embedding_model=embedding_model,
            vector_store_class=vector_store_class,
            vector_store_config=vector_store_config,
            api_key=api_key
        )
        self.retriever = DocumentRetriever(**(retrieval_params or {}))
        
        logger.info("RAGDocumentPipeline initialized with complete workflow")
    
    def create_rag_system(
        self,
        folder_path: Union[str, Path],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        recursive: bool = True,
        ignore_errors: bool = False,
        index_name: Optional[str] = None,
        save_index: bool = False,
        index_save_path: Optional[Union[str, Path]] = None,
        
    ) -> Dict[str, Any]:
        """
        Complete end-to-end RAG system creation from documents.
        
        Args:
            folder_path: Path to directory containing documents
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
            recursive: Whether to search subdirectories
            ignore_errors: Whether to continue if individual files fail
            index_name: Name for the vector index
            save_index: Whether to save the index to disk
            index_save_path: Path to save the index
           
        Returns:
            Dictionary with pipeline results and statistics
        """
        logger.info(f"Creating RAG system from: {folder_path}")
        
        try:
            # Step 1: Load documents
            logger.info("Step 1: Loading documents...")
            documents = self.document_processor.load_documents_from_directory(
                folder_path, recursive=recursive, ignore_errors=ignore_errors
            )
            
            if not documents:
                return {"success": False, "error": "No documents loaded"}
            
            doc_info = self.document_processor.get_document_info(documents)
            logger.info(f"Loaded {doc_info['total_documents']} documents")
            
            # Step 2: Split documents
            logger.info("Step 2: Splitting documents...")
            chunks = self.text_splitter.split_documents(
                documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            
            chunk_texts = [chunk.page_content for chunk in chunks]
            chunk_info = self.text_splitter.get_chunk_info(chunk_texts)
            logger.info(f"Created {chunk_info['total_chunks']} chunks")
            
            # Step 3: Create vector index
            logger.info("Step 3: Creating vector index...")
            vector_store = self.indexer.create_index(chunks, index_name=index_name)
            
            # Step 4: Setup retriever
            logger.info("Step 4: Setting up retriever...")
            self.retriever.set_vector_store(vector_store)
            
            # Step 5: Save index if requested
            if save_index and index_save_path:
                logger.info("Step 5: Saving index...")
                self.indexer.save_index(index_save_path)
            
            index_info = self.indexer.get_index_info()
            retrieval_info = self.retriever.get_retrieval_info()
            
            result = {
                "success": True,
                "document_info": doc_info,
                "chunk_info": chunk_info,
                "index_info": index_info,
                "retrieval_info": retrieval_info,
                "total_processing_time": "N/A"  # Could add timing if needed
            }
            
            logger.info("RAG system creation completed successfully!")
            return result
            
        except Exception as e:
            logger.error(f"Error creating RAG system: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def process_and_index_file(
        self,
        file_path: Union[str, Path],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        add_to_existing: bool = False,
      
    ) -> Dict[str, Any]:
        """
        Process single file and add to index.
        
        Args:
            file_path: Path to document file
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
            add_to_existing: Whether to add to existing index or create new
           
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing and indexing file: {file_path}")
        
        try:
            # Load and split document
            documents = self.document_processor.load_document(file_path)
            chunks = self.text_splitter.split_documents(
                documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            
            # Index documents
            if add_to_existing and self.indexer.vector_store is not None:
                doc_ids = self.indexer.add_documents(chunks)
                action = "added_to_existing"
            else:
                self.indexer.create_index(chunks)
                doc_ids = None
                action = "created_new_index"
            
            # Update retriever
            if self.indexer.vector_store:
                self.retriever.set_vector_store(self.indexer.vector_store)
            
            return {
                "success": True,
                "action": action,
                "chunks_processed": len(chunks),
                "document_ids": doc_ids
            }
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def query(
        self,
        query: str,
        search_type: str = 'similarity',
        k: int = 5,
        return_scores: bool = False,
        **retrieval_kwargs
    ) -> Union[List[Document], List[Tuple[Document, float]], Dict[str, Any]]:
        """
        Query the RAG system for relevant documents.
        
        Args:
            query: Search query string
            search_type: Type of search ('similarity', 'mmr', 'similarity_score_threshold')
            k: Number of documents to retrieve
            return_scores: Whether to return similarity scores
            **retrieval_kwargs: Additional retrieval parameters
            
        Returns:
            List of relevant documents or error information
        """
        if self.retriever.vector_store is None:
            return {"success": False, "error": "No vector store available. Create RAG system first."}
        
        try:
            if return_scores and search_type != 'similarity_score_threshold':
                results = self.retriever.similarity_search_with_score(
                    query, k=k, **retrieval_kwargs
                )
            else:
                results = self.retriever.retrieve(
                    query, search_type=search_type, k=k, **retrieval_kwargs
                )
            
            logger.info(f"Query '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def batch_query(
        self,
        queries: List[str],
        search_type: str = 'similarity',
        k: int = 5,
        **retrieval_kwargs
    ) -> List[List[Document]]:
        """
        Query the RAG system with multiple queries.
        
        Args:
            queries: List of search query strings
            search_type: Type of search to perform
            k: Number of documents to retrieve per query
            **retrieval_kwargs: Additional retrieval parameters
            
        Returns:
            List of document lists (one per query)
        """
        if self.retriever.vector_store is None:
            logger.error("No vector store available for batch query")
            return [[] for _ in queries]
        
        return self.retriever.batch_retrieve(
            queries, search_type=search_type, k=k, **retrieval_kwargs
        )
    
    def load_existing_index(
        self,
        index_path: Union[str, Path],
       
    ) -> bool:
        """
        Load an existing vector index.
        
        Args:
            index_path: Path to the saved index
            
            
        Returns:
            True if successful, False otherwise
        """
        try:
            vector_store = self.indexer.load_index(index_path)
            self.retriever.set_vector_store(vector_store)
            logger.info(f"Successfully loaded index from {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def save_current_index(
        self,
        index_path: Union[str, Path]
    ) -> bool:
        """
        Save the current vector index.
        
        Args:
            index_path: Path to save the index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.indexer.save_index(index_path)
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the RAG pipeline.
        
        Returns:
            Dictionary with pipeline status and statistics
        """
        return {
            "document_processor": {
                "loader_type": self.document_processor.document_loader_class.__name__,
                "supported_extensions": list(self.document_processor.supported_extensions)
            },
            "text_splitter": {
                "splitter_type": self.text_splitter.splitter_class.__name__,
                "default_params": self.text_splitter.default_params
            },
            "indexer": self.indexer.get_index_info(),
            "retriever": self.retriever.get_retrieval_info()
        }
    
    def reset_pipeline(self) -> None:
        """
        Reset the pipeline by clearing the vector store.
        """
        self.indexer.vector_store = None
        self.retriever.vector_store = None
        logger.info("Pipeline reset completed")