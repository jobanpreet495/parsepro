import logging
from pathlib import Path
from typing import List, Optional, Union, Type, Any, Dict

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.schema import Document
except ImportError as e:
    raise ImportError(
        "Required dependencies not found. Please install: "
        "pip install langchain langchain-community langchain-text-splitters "
        "langchain-openai faiss-cpu chromadb"
    ) from e


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    A professional document processor for RAG applications.
    
    Handles loading documents from files or directories with support for
    multiple document formats and loaders.
    
    Attributes:
        document_loader_class: The document loader class to use
        supported_extensions: Set of supported file extensions
    """
    
    # Default supported extensions for different loaders
    LOADER_EXTENSIONS = {
        PyPDFLoader: {'.pdf'},
        # Add more loaders and their extensions as needed
    }
    
    def __init__(self, document_loader: Type = PyPDFLoader):
        """
        Initialize the DocumentProcessor.
        
        Args:
            document_loader: Document loader class (default: PyPDFLoader)
        """
        self.document_loader_class = document_loader
        self.supported_extensions = self.LOADER_EXTENSIONS.get(
            document_loader, {'.pdf'}
        )
        
        logger.info(f"DocumentProcessor initialized with {document_loader.__name__}")
        logger.info(f"Supported extensions: {self.supported_extensions}")
    
    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a single document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file extension not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file extension: {file_path.suffix}. "
                f"Supported: {self.supported_extensions}"
            )
        
        try:
            loader = self.document_loader_class(str(file_path))
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} documents from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def load_documents_from_directory(
        self, 
        folder_path: Union[str, Path],
        recursive: bool = True,
        ignore_errors: bool = False
    ) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            folder_path: Path to the directory containing documents
            recursive: Whether to search subdirectories (default: True)
            ignore_errors: Whether to continue processing if individual files fail
            
        Returns:
            List of all loaded Document objects
            
        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Directory not found: {folder_path}")
        
        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        all_files = folder_path.glob(pattern)
        
        supported_files = [
            f for f in all_files 
            if f.is_file() and f.suffix.lower() in self.supported_extensions
        ]
        
        if not supported_files:
            logger.warning(f"No supported documents found in {folder_path}")
            return []
        
        logger.info(f"Found {len(supported_files)} supported documents in {folder_path}")
        
        all_documents = []
        failed_files = []
        
        for file_path in supported_files:
            try:
                documents = self.load_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                failed_files.append((file_path, str(e)))
                if not ignore_errors:
                    raise
                logger.warning(f"Failed to load {file_path}: {str(e)}")
        
        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files")
        
        logger.info(f"Successfully loaded {len(all_documents)} total documents")
        return all_documents
    


    def get_document_info(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get summary information about loaded documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {"total_documents": 0, "total_characters": 0, "average_length": 0}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_length = total_chars / len(documents)
        
        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "average_length": round(avg_length, 2),
            "sources": list(set(doc.metadata.get('source', 'Unknown') for doc in documents))
        }
    
   
class TextSplitter:
    """
    A professional text splitter for RAG applications.
    
    Handles splitting documents into chunks with customizable parameters
    and support for different splitting strategies.
    
    Attributes:
        splitter_class: The text splitter class to use
        default_params: Default parameters for the splitter
    """
    
    # Default parameters for different splitters
    DEFAULT_PARAMS = {
        RecursiveCharacterTextSplitter: {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'separators': ["\n\n", "\n", " ", ""],
            'length_function': len,
        }
    }
    
    def __init__(
        self, 
        text_splitter: Type = RecursiveCharacterTextSplitter,
        **default_kwargs
    ):
        """
        Initialize the TextSplitter.
        
        Args:
            text_splitter: Text splitter class (default: RecursiveCharacterTextSplitter)
            **default_kwargs: Default parameters to override
        """
        self.splitter_class = text_splitter
        
        # Get default parameters and update with any provided kwargs
        self.default_params = self.DEFAULT_PARAMS.get(
            text_splitter, 
            {'chunk_size': 1000, 'chunk_overlap': 200}
        ).copy()
        self.default_params.update(default_kwargs)
        
        logger.info(f"TextSplitter initialized with {text_splitter.__name__}")
        logger.info(f"Default parameters: {self.default_params}")
    
    def split_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        
    ) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects to split
            chunk_size: Size of each chunk (default: from class defaults)
            chunk_overlap: Overlap between chunks (default: from class defaults)
            separators: List of separators to use (default: from class defaults)
          
            
        Returns:
            List of split Document objects
        """
        if not documents:
            logger.warning("No documents provided for splitting")
            return []
        
        # Prepare parameters
        params = self.default_params.copy()
        
        if chunk_size is not None:
            params['chunk_size'] = chunk_size
        if chunk_overlap is not None:
            params['chunk_overlap'] = chunk_overlap
        if separators is not None:
            params['separators'] = separators
        
        
        
        try:
            # Create splitter instance
            splitter = self.splitter_class(**params)
            
            # Split documents
            split_docs = splitter.split_documents(documents)
            
            logger.info(
                f"Split {len(documents)} documents into {len(split_docs)} chunks "
                f"(chunk_size: {params['chunk_size']}, overlap: {params['chunk_overlap']})"
            )
            
            return split_docs
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def split_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
       
    ) -> List[str]:
        """
        Split raw text into chunks.
        
        Args:
            text: Text string to split
            chunk_size: Size of each chunk (default: from class defaults)
            chunk_overlap: Overlap between chunks (default: from class defaults)
            separators: List of separators to use (default: from class defaults)
          
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            logger.warning("Empty text provided for splitting")
            return []
        
        # Prepare parameters
        params = self.default_params.copy()
        
        if chunk_size is not None:
            params['chunk_size'] = chunk_size
        if chunk_overlap is not None:
            params['chunk_overlap'] = chunk_overlap
        if separators is not None:
            params['separators'] = separators
        
        try:
            # Create splitter instance
            splitter = self.splitter_class(**params)
            
            # Split text
            chunks = splitter.split_text(text)
            
            logger.info(
                f"Split text ({len(text)} chars) into {len(chunks)} chunks "
                f"(chunk_size: {params['chunk_size']}, overlap: {params['chunk_overlap']})"
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            raise


    def get_chunk_info(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Get summary information about text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {"total_chunks": 0, "total_characters": 0, "average_length": 0}
        
        chunk_lengths = [len(chunk) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_lengths),
            "average_length": round(sum(chunk_lengths) / len(chunks), 2),
            "min_length": min(chunk_lengths),
            "max_length": max(chunk_lengths)
        }
    
   












