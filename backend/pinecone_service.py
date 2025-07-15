import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

class PineconeService:
    def __init__(self, user_id: str = "default"):
        """
        Initialize Pinecone service with user-specific settings.
        
        Args:
            user_id: Unique identifier for the user (used for namespacing)
        """
        self.user_id = user_id
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize Pinecone
        self._init_pinecone()
        
    def _init_pinecone(self):
        """Initialize Pinecone connection and index"""
        # Get config from environment
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENV", "gcp-starter")
        index_name = f"{os.getenv('PINECONE_INDEX_NAME', 'knowledge-base')}-{self.user_id}"
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
            
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        
        # Create index if it doesn't exist
        if index_name not in [index.name for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Free tier supported region
                )
            )
        
        # Connect to the index
        self.index = self.pc.Index(index_name)
    
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
