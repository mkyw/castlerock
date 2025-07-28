import os
import logging
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec, NotFoundException

# Local imports
from utils.index_utils import generate_index_name, get_user_indices

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PineconeService:
    def __init__(self, user_id: str, index_name: Optional[str] = None, display_name: Optional[str] = None):
        """
        Initialize Pinecone service with user-specific settings.
        
        Args:
            user_id: Unique identifier for the user (email)
            index_name: Optional full name for the index (if None, will generate one)
            display_name: Optional display name for the index (used if index_name is None)
        """
        self.user_id = user_id
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # If index_name is provided, use it directly, otherwise generate one
        if index_name is None:
            self.index_name, _ = generate_index_name(user_id, display_name or 'default')
        else:
            self.index_name = index_name
        
        # Initialize Pinecone connection
        self._init_pinecone()
    
    @classmethod
    def get_user_indices(cls, user_id: str) -> List[Dict[str, Any]]:
        """Get all indices for a user.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            List of index information dictionaries
        """
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
            
        pc = Pinecone(api_key=api_key)
        user_indices = []
        
        # Get all indices and filter for this user's indices
        for index in pc.list_indexes():
            # Match indices that follow the pattern: castlerock-{user_hash}-{index_name}
            if index.name.startswith(f"castlerock-{get_user_hash(user_id)}-"):
                # Extract the custom index name part
                index_name_parts = index.name.split('-')
                custom_name = '-'.join(index_name_parts[2:]) if len(index_name_parts) > 2 else 'default'
                
                user_indices.append({
                    'name': index.name,
                    'display_name': custom_name,
                    'dimension': index.dimension,
                    'metric': index.metric,
                    'status': index.status.state,
                    'created_at': getattr(index, 'created_at', None)
                })
                
        return user_indices
        
    def _generate_index_name(self, custom_name: Optional[str] = None) -> str:
        """Generate a standard index name for a user.
        
        Args:
            custom_name: Optional custom name for the index
            
        Returns:
            Formatted index name
        """
        # Use the utility function to generate the index name
        index_name, _ = generate_index_name(self.user_id, custom_name or 'default')
        return index_name
        
    def _init_pinecone(self):
        """Initialize Pinecone connection and index"""
        # Get config from environment
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
            
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        
        # Check if index exists, create if it doesn't
        logger.info(f"Initializing Pinecone service for index: {self.index_name}")
        self._ensure_index_exists()
        
        # Connect to the index
        self.index = self.pc.Index(self.index_name)
    
    def _ensure_index_exists(self):
        """Ensure the index exists, create it if it doesn't"""
        try:
            # First, try to directly describe the index - this is faster than listing all indices
            try:
                status = self.pc.describe_index(self.index_name)
                logger.info(f"Index {self.index_name} already exists")
                return True
            except Exception as e:
                # If the index doesn't exist, we'll get an exception
                if "not found" not in str(e).lower() and "resource not found" not in str(e).lower():
                    # If it's some other error, re-raise it
                    raise
                
                # Index doesn't exist, create it
                logger.info(f"Index {self.index_name} not found, creating it")
                self._create_index()
                return True
                
        except Exception as e:
            logger.error(f"Error in _ensure_index_exists: {str(e)}")
            # If we get here, there was an error with describe_index or _create_index
            # Let's try to be more specific about the error
            if "already exists" in str(e).lower() or "already_exists" in str(e).lower():
                logger.info(f"Index {self.index_name} already exists (from exception)")
                return True
            raise
    
    def _create_index(self):
        """Create a new index for the user"""
        try:
            # Try to create the index directly
            logger.info(f"Creating new index: {self.index_name}")
            
            # Create the index with the specified name
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Free tier supported region
                )
            )
            
            # Wait for the index to be ready
            import time
            max_attempts = 60  # 60 seconds max wait
            attempts = 0
            
            while attempts < max_attempts:
                try:
                    status = self.pc.describe_index(self.index_name)
                    if status.status.ready:
                        logger.info(f"Index {self.index_name} is ready")
                        return
                    time.sleep(1)
                    attempts += 1
                except Exception as e:
                    logger.warning(f"Error checking index status (attempt {attempts + 1}/{max_attempts}): {e}")
                    time.sleep(1)
                    attempts += 1
            
            logger.warning(f"Timed out waiting for index {self.index_name} to be ready")
            
        except Exception as e:
            # Handle the case where index already exists
            if "already exists" in str(e).lower() or "already_exists" in str(e).lower():
                logger.info(f"Index {self.index_name} already exists, skipping creation")
                return
            # Re-raise any other exceptions
            logger.error(f"Error creating index {self.index_name}: {e}")
            raise
    
    def delete_index(self):
        """Delete the current index"""
        try:
            self.pc.delete_index(self.index_name)
            return True
        except Exception as e:
            print(f"Error deleting index {self.index_name}: {str(e)}")
            return False
    
    def upsert_document(self, text: str, metadata: Dict[str, Any]) -> None:
        """
        Process and upsert a document into Pinecone.
        
        Args:
            text: Document text content
            metadata: Additional metadata including 'source' and other relevant info
        """
        import uuid
        
        # Generate a unique ID for this document to ensure vector IDs are unique
        doc_uuid = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for brevity
        source = metadata.get('source', 'doc')
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks)
        
        # Prepare vectors for Pinecone with unique IDs
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create a unique vector ID using source, chunk index, and document UUID
            vector_id = f"{source}-{doc_uuid}-{i}"
            vectors.append((
                vector_id,
                embedding.tolist(),
                {"text": chunk, **metadata, "vector_id": vector_id}
            ))
        
        # Upsert in batches of 100
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i+100]
            self.index.upsert(vectors=batch)
    
    async def query_batch(self, queries: List[str], k: int = 5, batch_size: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Query the Pinecone index with multiple queries in parallel batches.
        
        Args:
            queries: List of search queries
            k: Number of results to return per query
            batch_size: Number of queries to process in parallel
            
        Returns:
            List of lists containing matching documents with metadata for each query
        """
        import asyncio
        from typing import List, Dict, Any
        
        async def process_batch(batch_queries: List[str]) -> List[List[Dict[str, Any]]]:
            # Generate embeddings for the batch
            batch_embeddings = self.embedding_model.encode(batch_queries)
            
            # Process queries 
            results = []
            for query, embedding in zip(batch_queries, batch_embeddings):
                try:
                    # Use synchronous query instead of query_async
                    result = self.index.query(
                        vector=embedding.tolist(),
                        top_k=k,
                        include_metadata=True
                    )
                    
                    formatted_results = [
                        {
                            "text": match.metadata["text"],
                            "source": match.metadata.get("source", "Unknown"),
                            "score": match.score,
                            "query": query
                        }
                        for match in result.matches
                    ]
                    results.append(formatted_results)
                except Exception as e:
                    print(f"Error processing query '{query}': {e}")
                    results.append([])
            
            return results
        
        # Process queries in batches
        all_results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_results = await process_batch(batch)
            all_results.extend(batch_results)
        
        return all_results
    
    async def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the Pinecone index with a single query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of matching documents with metadata
        """
        results = await self.query_batch([query], k=k)
        return results[0] if results else []
    
    def delete_index(self) -> bool:
        """Delete the user's Pinecone index"""
        try:
            self.pc.delete_index(self.index.name)
            return True
        except Exception as e:
            print(f"Error deleting index: {e}")
            return False
