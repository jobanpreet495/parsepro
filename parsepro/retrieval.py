from langchain.vectorstores.base import VectorStore
from typing import List, Optional, Union, Any, Dict , Tuple
from langchain.schema import Document
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Professional document retriever for RAG applications.
    
    Handles similarity search and document retrieval with support for
    multiple similarity metrics and retrieval strategies.
    
    Attributes:
        vector_store: The vector store instance for retrieval
        default_retrieval_params: Default parameters for retrieval
    """
    
    # Default retrieval parameters
    DEFAULT_RETRIEVAL_PARAMS = {
        'k': 5,
        'score_threshold': None,
        'fetch_k': 20,
        'lambda_mult': 0.5,
        'search_type': 'similarity'
    }
    
    # Available search types
    SEARCH_TYPES = {
        'similarity': 'Standard similarity search',
        'mmr': 'Maximum Marginal Relevance search',
        'similarity_score_threshold': 'Similarity search with score threshold'
    }
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        **default_params
    ):
        """
        Initialize the DocumentRetriever.
        
        Args:
            vector_store: Vector store instance for retrieval
            **default_params: Default parameters to override
        """
        self.vector_store = vector_store
        self.default_retrieval_params = self.DEFAULT_RETRIEVAL_PARAMS.copy()
        self.default_retrieval_params.update(default_params)
        
        logger.info("DocumentRetriever initialized")
        if vector_store:
            logger.info(f"Vector store: {type(vector_store).__name__}")
        logger.info(f"Default retrieval params: {self.default_retrieval_params}")
    
    def set_vector_store(self, vector_store: VectorStore) -> None:
        """
        Set the vector store for retrieval.
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store
        logger.info(f"Vector store set: {type(vector_store).__name__}")
    
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
      
    ) -> List[Document]:
        """
        Perform similarity search for documents.
        
        Args:
            query: Search query string
            k: Number of documents to retrieve (default: from class defaults)
            
            
        Returns:
            List of relevant Document objects
            
        Raises:
            ValueError: If no vector store is set
        """
        if self.vector_store is None:
            raise ValueError("No vector store set for retrieval")
        
        k = k or self.default_retrieval_params['k']
        
        try:
            logger.info(f"Performing similarity search for: '{query}' (k={k})")
            documents = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
      
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with similarity scores.
        
        Args:
            query: Search query string
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            
            
        Returns:
            List of tuples (Document, similarity_score)
        """
        if self.vector_store is None:
            raise ValueError("No vector store set for retrieval")
        
        k = k or self.default_retrieval_params['k']
        score_threshold = score_threshold or self.default_retrieval_params['score_threshold']
        
        try:
            logger.info(f"Performing similarity search with scores for: '{query}' (k={k})")
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Apply score threshold if specified
            if score_threshold is not None:
                results = [(doc, score) for doc, score in results if score >= score_threshold]
                logger.info(f"Applied score threshold {score_threshold}, {len(results)} documents remaining")
            
            logger.info(f"Retrieved {len(results)} documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {str(e)}")
            raise
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        
    ) -> List[Document]:
        """
        Perform Maximum Marginal Relevance (MMR) search.
        
        Args:
            query: Search query string
            k: Number of documents to return
            fetch_k: Number of documents to fetch initially
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
          
            
        Returns:
            List of relevant Document objects with diversity
        """
        if self.vector_store is None:
            raise ValueError("No vector store set for retrieval")
        
        k = k or self.default_retrieval_params['k']
        fetch_k = fetch_k or self.default_retrieval_params['fetch_k']
        lambda_mult = lambda_mult or self.default_retrieval_params['lambda_mult']
        
        try:
            logger.info(f"Performing MMR search for: '{query}' (k={k}, fetch_k={fetch_k}, lambda={lambda_mult})")
            
            if hasattr(self.vector_store, 'max_marginal_relevance_search'):
                documents = self.vector_store.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
                )
            else:
                logger.warning("MMR search not supported by vector store, falling back to similarity search")
                documents = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"Retrieved {len(documents)} documents via MMR")
            return documents
            
        except Exception as e:
            logger.error(f"Error in MMR search: {str(e)}")
            raise
    
    def retrieve(
        self,
        query: str,
        search_type: str = 'similarity',
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Universal retrieval method with configurable search strategy.
        
        Args:
            query: Search query string
            search_type: Type of search ('similarity', 'mmr', 'similarity_score_threshold')
            k: Number of documents to retrieve
            score_threshold: Score threshold for filtering results
            fetch_k: Number of documents to fetch initially (for MMR)
            lambda_mult: Diversity parameter (for MMR)
            **kwargs: Additional parameters for search
            
        Returns:
            List of Document objects or tuples (Document, score)
        """
        if search_type not in self.SEARCH_TYPES:
            raise ValueError(f"Invalid search type: {search_type}. Available: {list(self.SEARCH_TYPES.keys())}")
        
        logger.info(f"Retrieving documents using {search_type} search")
        
        if search_type == 'similarity':
            return self.similarity_search(query, k=k)
        elif search_type == 'mmr':
            return self.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult,
            )
        elif search_type == 'similarity_score_threshold':
            return self.similarity_search_with_score(
                query, k=k, score_threshold=score_threshold, 
            )
        else:
            raise ValueError(f"Search type {search_type} not implemented")
    
    def batch_retrieve(
        self,
        queries: List[str],
        search_type: str = 'similarity',
        k: Optional[int] = None,
      
    ) -> List[List[Document]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of search query strings
            search_type: Type of search to perform
            k: Number of documents to retrieve per query
           
            
        Returns:
            List of document lists (one per query)
        """
        logger.info(f"Batch retrieving for {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries):
            try:
                query_results = self.retrieve(query, search_type=search_type, k=k)
                # Ensure we return Document objects only for batch retrieval
                if query_results and isinstance(query_results[0], tuple):
                    query_results = [doc for doc, score in query_results]
                results.append(query_results)
                logger.debug(f"Query {i+1}/{len(queries)}: {len(query_results)} documents")
            except Exception as e:
                logger.error(f"Error retrieving for query {i+1}: {e}")
                results.append([])
        
        logger.info(f"Batch retrieval completed for {len(queries)} queries")
        return results
    
    def get_retrieval_info(self) -> Dict[str, Any]:
        """
        Get information about retrieval configuration.
        
        Returns:
            Dictionary with retrieval configuration
        """
        return {
            "vector_store_available": self.vector_store is not None,
            "vector_store_type": type(self.vector_store).__name__ if self.vector_store else None,
            "default_params": self.default_retrieval_params,
            "available_search_types": self.SEARCH_TYPES
        }