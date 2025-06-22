import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Set, Dict, Optional, Any, AsyncGenerator, Tuple
import tiktoken
import os
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from collections import deque
import signal

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

# Configuration
CONFIG = {
    "max_concurrent_requests": 5,  # Number of concurrent HTTP requests
    "max_embedding_workers": 3,     # Number of concurrent embedding workers
    "request_timeout": 10,          # Request timeout in seconds
    "batch_size": 10,               # Number of documents to process in a batch
    "save_interval": 5,             # Save vector store every N batches
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

@dataclass
class CrawlResult:
    url: str
    content: Optional[str]
    links: List[str]
    error: Optional[str] = None

# Load environment variables
load_dotenv()

class KBScraper:
    def __init__(self, base_url: str = "https://kb.wisc.edu/", max_pages: int = 100, persist_dir: str = "faiss_index"):
        """
        Initialize the KB Scraper with FAISS vector store
        
        Args:
            base_url: The base URL to start scraping from
            max_pages: Maximum number of pages to scrape (0 for unlimited)
            persist_dir: Directory to store the FAISS index and metadata
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.persist_dir = Path(persist_dir)
        self.visited_urls: Set[str] = set()
        self.pages_to_visit = deque([base_url])
        self.documents_processed = 0
        self.batches_processed = 0
        
        # Initialize OpenAI embeddings
        self.embeddings: Embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.vectorstore: Optional[FAISS] = None
        
        # Create persist directory if it doesn't exist
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize text splitter with appropriate chunk size and overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Queue for passing documents to embedding workers
        self.doc_queue: asyncio.Queue[Optional[Document]] = asyncio.Queue()
        
        # Event to signal completion
        self.done_event = asyncio.Event()
        
        # Load or create vector store
        self._load_vectorstore()
        
        # Track running tasks
        self.tasks: Set[asyncio.Task] = set()
        
        # For graceful shutdown
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        if self.shutdown_requested:
            print("\nForce shutdown requested. Exiting immediately.")
            import sys
            sys.exit(1)
            
        print("\nShutdown requested. Finishing current tasks...")
        self.shutdown_requested = True
        self.done_event.set()
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to the same domain"""
        if not url or not isinstance(url, str):
            return False
            
        try:
            parsed = urlparse(url)
            base_parsed = urlparse(self.base_url)
            return (parsed.netloc == base_parsed.netloc and 
                   parsed.scheme in ['http', 'https'] and
                   not any(ext in url.lower() for ext in ['.pdf', '.jpg', '.jpeg', '.png', '.gif']))
        except Exception:
            return False
    
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> CrawlResult:
        """Fetch a single page and extract content and links"""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=CONFIG["request_timeout"])) as response:
                if response.status != 200:
                    return CrawlResult(url, None, [], f"HTTP {response.status}")
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract main content (customize selectors as needed)
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                content = ' '.join(main_content.stripped_strings) if main_content else None
                
                # Extract links
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(url, href)
                    if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                        links.append(absolute_url)
                
                return CrawlResult(url, content, links)
                
        except Exception as e:
            return CrawlResult(url, None, [], str(e))
    
    async def process_document(self, url: str, content: str) -> None:
        """Process a single document and add it to the queue"""
        try:
            # Split the content into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create documents for each chunk
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": url,
                        "chunk": i,
                        "total_chunks": len(chunks),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                await self.doc_queue.put(doc)
                self.documents_processed += 1
                
                if self.documents_processed % 10 == 0:
                    print(f"Processed {self.documents_processed} chunks...")
                    
        except Exception as e:
            print(f"Error processing {url}: {e}")
    
    def _load_vectorstore(self) -> None:
        """Load existing FAISS vector store if available"""
        index_path = self.persist_dir / "index.faiss"
        if index_path.exists():
            print("Loading existing FAISS index...")
            try:
                self.vectorstore = FAISS.load_local(
                    folder_path=str(self.persist_dir),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                    index_name="index"
                )
                print(f"Loaded {self.vectorstore.index.ntotal} vectors from existing index.")
                return
            except Exception as e:
                print(f"Error loading existing index: {e}")
                print("Creating a new index...")
        
        # Create a dummy document to initialize the index
        dummy_doc = Document(
            page_content="dummy",
            metadata={"source": "dummy", "chunk": 0, "total_chunks": 1}
        )
        self.vectorstore = FAISS.from_documents(
            documents=[dummy_doc],
            embedding=self.embeddings
        )
        # Remove the dummy document
        self.vectorstore.delete([0])
        print("Created a new empty FAISS index.")
    
    async def save_vectorstore_async(self) -> None:
        """Save the FAISS index asynchronously"""
        if self.vectorstore:
            # Save to a temporary file first, then rename to be atomic
            temp_path = self.persist_dir / "index.tmp"
            self.vectorstore.save_local(folder_path=str(temp_path), index_name="index")
            
            # Rename temp directory to final name
            final_path = self.persist_dir / "index.faiss"
            if final_path.exists():
                import shutil
                shutil.rmtree(final_path)
            temp_path.rename(final_path)
            
            print(f"Saved vector store with {self.vectorstore.index.ntotal} vectors.")
    
    async def embedding_worker(self, worker_id: int) -> None:
        """Worker that processes documents and adds them to the vector store"""
        print(f"Embedding worker {worker_id} started")
        
        batch = []
        last_save = time.time()
        
        while True:
            try:
                # Get a document with timeout to allow for periodic saves
                try:
                    doc = await asyncio.wait_for(self.doc_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    doc = None
                
                # Process document if available
                if doc is not None:
                    batch.append(doc)
                    self.doc_queue.task_done()
                
                # Process batch if we have enough documents or it's time to save
                current_time = time.time()
                if (batch and (len(batch) >= CONFIG["batch_size"] or 
                             current_time - last_save >= CONFIG["save_interval"] * 60)):
                    try:
                        if self.vectorstore is None:
                            self.vectorstore = FAISS.from_documents(
                                documents=batch,
                                embedding=self.embeddings
                            )
                        else:
                            self.vectorstore.add_documents(batch)
                        
                        self.batches_processed += 1
                        print(f"Worker {worker_id}: Processed batch {self.batches_processed} "
                              f"(total vectors: {self.vectorstore.index.ntotal})")
                        
                        # Save periodically
                        if current_time - last_save >= CONFIG["save_interval"] * 60:
                            await self.save_vectorstore_async()
                            last_save = current_time
                        
                        batch = []
                        
                    except Exception as e:
                        print(f"Error in embedding worker {worker_id}: {e}")
                
                # Check if we should exit
                if self.shutdown_requested and self.doc_queue.empty():
                    if batch:  # Process any remaining documents
                        try:
                            if self.vectorstore is None:
                                self.vectorstore = FAISS.from_documents(
                                    documents=batch,
                                    embedding=self.embeddings
                                )
                            else:
                                self.vectorstore.add_documents(batch)
                            print(f"Worker {worker_id}: Processed final batch with {len(batch)} documents")
                        except Exception as e:
                            print(f"Error in final batch processing: {e}")
                    break
                        
            except Exception as e:
                print(f"Error in embedding worker {worker_id}: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
        
        print(f"Embedding worker {worker_id} finished")
    
    async def crawl_worker(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> None:
        """Worker that crawls pages and processes them"""
        while not self.shutdown_requested:
            try:
                # Get next URL to process
                try:
                    url = self.pages_to_visit.popleft()
                except IndexError:
                    # No more URLs to process
                    if not self.done_event.is_set():
                        self.done_event.set()
                    break
                
                # Skip if already visited or invalid
                if url in self.visited_urls or not self.is_valid_url(url):
                    continue
                    
                # Check max pages limit
                if self.max_pages > 0 and len(self.visited_urls) >= self.max_pages:
                    print(f"Reached maximum page limit ({self.max_pages}). Stopping...")
                    self.shutdown_requested = True
                    break
                
                # Add to visited set
                self.visited_urls.add(url)
                
                # Process the page with rate limiting
                async with semaphore:
                    print(f"Crawling: {url}")
                    result = await self.fetch_page(session, url)
                    
                    # Add new links to the queue
                    if result.links:
                        for link in result.links:
                            if link not in self.visited_urls and link not in self.pages_to_visit:
                                self.pages_to_visit.append(link)
                    
                    # Process the page content if available
                    if result.content:
                        await self.process_document(result.url, result.content)
                    
                    # Small delay between requests to be nice to the server
                    await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"Error in crawl worker: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def crawl(self) -> None:
        """Asynchronously crawl the website starting from base_url"""
        print(f"Starting crawl of {self.base_url}")
        print(f"Concurrent requests: {CONFIG['max_concurrent_requests']}")
        print(f"Embedding workers: {CONFIG['max_embedding_workers']}")
        
        # Create aiohttp session
        connector = aiohttp.TCPConnector(limit=CONFIG["max_concurrent_requests"])
        headers = {"User-Agent": CONFIG["user_agent"]}
        
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            # Start embedding workers
            embedding_workers = []
            for i in range(CONFIG["max_embedding_workers"]):
                task = asyncio.create_task(self.embedding_worker(i))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
                embedding_workers.append(task)
            
            # Start crawl workers
            semaphore = asyncio.Semaphore(CONFIG["max_concurrent_requests"])
            crawl_workers = []
            for _ in range(CONFIG["max_concurrent_requests"]):
                task = asyncio.create_task(self.crawl_worker(session, semaphore))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
                crawl_workers.append(task)
            
            # Wait for all workers to complete
            try:
                await asyncio.gather(*crawl_workers)
                print("Crawl workers finished. Waiting for embedding workers to complete...")
                
                # Signal embedding workers to finish
                for _ in range(CONFIG["max_embedding_workers"]):
                    await self.doc_queue.put(None)
                
                await asyncio.gather(*embedding_workers)
                
            except asyncio.CancelledError:
                print("Crawl was cancelled. Cleaning up...")
                self.shutdown_requested = True
                
                # Cancel all tasks
                for task in self.tasks:
                    task.cancel()
                
                # Wait for tasks to complete
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Final save
            if self.vectorstore:
                await self.save_vectorstore_async()
            
            print("\nCrawl completed successfully!")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search the vector store for similar documents
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of matching documents with similarity scores
        """
        if self.vectorstore is None:
            print("No documents in vector store. Please crawl some pages first.")
            return []
            
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_vector_count(self) -> int:
        """Get the number of vectors in the store"""
        if self.vectorstore is None:
            return 0
        return self.vectorstore.index.ntotal


async def main():
    # Initialize the scraper
    scraper = KBScraper(
        base_url="https://kb.wisc.edu/",
        max_pages=100,  # Set to 0 for unlimited
        persist_dir="kb_faiss_index"
    )
    
    try:
        # Start the crawl
        await scraper.crawl()
        
        # Display summary
        print("\n" + "="*50)
        print(f"Crawl completed successfully!")
        print(f"Pages crawled: {len(scraper.visited_urls)}")
        print(f"Document chunks processed: {scraper.documents_processed}")
        print(f"Batches processed: {scraper.batches_processed}")
        
        if scraper.vectorstore:
            vector_count = scraper.vectorstore.index.ntotal if hasattr(scraper.vectorstore, 'index') else 0
            print(f"Total vectors in store: {vector_count}")
            
            # Example search
            query = "How do I reset my password?"
            print(f"\nSearching for: {query}")
            try:
                results = scraper.vectorstore.similarity_search_with_score(query, k=3)
                for i, (doc, score) in enumerate(results, 1):
                    print(f"\nResult {i} (Score: {score:.3f}):")
                    print(f"Source: {doc.metadata['source']}")
                    print(f"Content: {doc.page_content[:200]}...")
            except Exception as e:
                print(f"Error during search: {e}")
        
    except asyncio.CancelledError:
        print("\nCrawling was cancelled.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup
        if scraper.vectorstore:
            try:
                await scraper.save_vectorstore_async()
                print("\nVector store saved successfully.")
            except Exception as e:
                print(f"Error saving vector store: {e}")
        
        print("\nExiting...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
