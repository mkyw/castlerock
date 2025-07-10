"""
Knowledge Base RAG System

This module provides functionality to create a knowledge base from websites or PDFs,
and query it using RAG (Retrieval-Augmented Generation) with Gemini.
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Set, Dict, Optional, Any, AsyncGenerator, Tuple, Union, BinaryIO
import tiktoken
import os
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from collections import deque
import signal
import hashlib
import tempfile
import shutil

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class CrawlResult:
    """Represents the result of crawling a single URL."""
    url: str
    content: Optional[str]
    links: List[str]
    error: Optional[str] = None

class KBRAGSystem:
    """
    A system for creating and querying a knowledge base from websites or PDFs.
    """
    
    def __init__(
        self, 
        source: Optional[Union[str, List[str], BinaryIO]] = None,
        source_type: str = 'website',  # 'website' or 'pdf'
        max_pages: int = 1000,
        persist_dir: str = "faiss_index",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the KB RAG System.
        
        Args:
            source: The source to process - can be a URL, list of URLs, or file-like object
            source_type: Type of source - 'website' or 'pdf'
            max_pages: Maximum number of pages to process (for websites)
            persist_dir: Directory to store the FAISS index and metadata
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        """
        self.source = source
        self.source_type = source_type
        self.max_pages = max_pages
        self.persist_dir = Path(persist_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize state
        self.visited_urls: Set[str] = set()
        self.pages_to_visit = deque()
        self.documents_processed = 0
        self.batches_processed = 0
        self._shutdown_lock = asyncio.Lock()
        self._save_lock = asyncio.Lock()
        
        # Initialize embeddings and vector store
        self.embeddings = self._init_embeddings()
        self.vectorstore: Optional[FAISS] = None
        
        # Create persist directory if it doesn't exist
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Queues and tasks
        self.doc_queue: asyncio.Queue[Optional[Document]] = asyncio.Queue(maxsize=1000)
        self.tasks: Set[asyncio.Task] = set()
        self.embedding_workers: List[asyncio.Task] = []
        
        # Shutdown handling
        self.shutdown_requested = False
        self._is_closed = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Load or create vector store
        self._load_vectorstore()
    
    def _init_embeddings(self) -> Embeddings:
        """Initialize HuggingFace embeddings with proper cleanup."""
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    
    def _load_vectorstore(self) -> None:
        """Load or create the FAISS vector store."""
        index_path = self.persist_dir / "index.faiss"
        if index_path.exists():
            print("Loading existing vector store...")
            self.vectorstore = FAISS.load_local(
                self.persist_dir, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Loaded {self.vectorstore.index.ntotal} vectors")
        else:
            print("Creating new vector store...")
            self.vectorstore = FAISS.from_documents(
                [Document(page_content="")],  # Empty document to initialize
                self.embeddings
            )
    
    async def process_website(self, url: str) -> None:
        """Process a single website URL."""
        self.pages_to_visit.append(url)
        self.visited_urls.add(url)
        
        async with aiohttp.ClientSession() as session:
            while self.pages_to_visit and len(self.visited_urls) < self.max_pages:
                if self.shutdown_requested:
                    break
                    
                current_url = self.pages_to_visit.popleft()
                print(f"Processing: {current_url}")
                
                try:
                    async with session.get(current_url) as response:
                        if response.status == 200:
                            content = await response.text()
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Extract main content (customize based on website structure)
                            main_content = soup.find('main') or soup.find('article') or soup
                            text = main_content.get_text(separator=' ', strip=True)
                            
                            # Create document
                            doc = Document(
                                page_content=text,
                                metadata={"source": current_url, "type": "website"}
                            )
                            
                            # Split and add to queue
                            await self._process_document(doc)
                            
                            # Extract and queue new links
                            base_domain = f"{urlparse(current_url).scheme}://{urlparse(current_url).netloc}"
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                full_url = urljoin(base_domain, href)
                                if (
                                    full_url.startswith(base_domain) and 
                                    full_url not in self.visited_urls and
                                    not any(ext in full_url.lower() for ext in ['.pdf', '.jpg', '.png', '.gif', '.zip', '.tar', '.gz'])
                                ):
                                    self.visited_urls.add(full_url)
                                    self.pages_to_visit.append(full_url)
                                    
                except Exception as e:
                    print(f"Error processing {current_url}: {str(e)}")
    
    async def process_pdf(self, file_path: Union[str, BinaryIO]) -> None:
        """Process a PDF file."""
        temp_file = None
        
        try:
            # If it's a file-like object, save to temp file
            if hasattr(file_path, 'read'):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                temp_file.write(file_path.read())
                temp_file.close()
                file_path = temp_file.name
            
            # Load and process PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            for page in pages:
                await self._process_document(page)
                
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            
        finally:
            # Clean up temp file if created
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    async def _process_document(self, doc: Document) -> None:
        """Process a single document (split, embed, add to vector store)."""
        if not doc.page_content.strip():
            return
            
        # Split document
        chunks = self.text_splitter.split_documents([doc])
        
        # Add to vector store
        if chunks:
            if not self.vectorstore:
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vectorstore.add_documents(chunks)
            
            self.documents_processed += len(chunks)
            self.batches_processed += 1
            
            # Save periodically
            if self.batches_processed % 5 == 0:
                await self.save_vectorstore()
    
    async def save_vectorstore(self) -> None:
        """Save the vector store to disk."""
        if self.vectorstore:
            async with self._save_lock:
                self.vectorstore.save_local(self.persist_dir)
    
    async def query(self, query: str, k: int = 5) -> List[Document]:
        """Query the knowledge base."""
        if not self.vectorstore:
            raise ValueError("No vector store available. Please load or create one first.")
            
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.vectorstore.similarity_search(query, k=k)
        )
    
    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals."""
        print("\nShutdown requested, cleaning up...")
        self.shutdown_requested = True
        
        # Save vector store before exiting
        asyncio.create_task(self._async_shutdown())
    
    async def _async_shutdown(self) -> None:
        """Perform async cleanup."""
        if not self._is_closed:
            self._is_closed = True
            await self.save_vectorstore()
            print("Cleanup complete. Exiting...")
            
    async def close(self) -> None:
        """Close the system and clean up resources."""
        await self._async_shutdown()

# Example usage
async def main():
    # Example 1: Process a website
    # rag = KBRAGSystem(source="https://example.com", source_type='website')
    # await rag.process_website("https://example.com")
    
    # Example 2: Process a PDF file
    # with open('document.pdf', 'rb') as f:
    #     rag = KBRAGSystem(source=f, source_type='pdf')
    #     await rag.process_pdf(f)
    
    # Query example
    # results = await rag.query("What is the main topic?")
    # for doc in results:
    #     print(f"Source: {doc.metadata['source']}")
    #     print(doc.page_content[:200] + "...\n")
    
    pass

if __name__ == "__main__":
    asyncio.run(main())
