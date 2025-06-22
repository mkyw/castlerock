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
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
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
    def __init__(self, base_url: str = "https://kb.wisc.edu/", max_pages: int = 1000000, persist_dir: str = "faiss_index"):
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
        self._shutdown_lock = asyncio.Lock()
        self._save_lock = asyncio.Lock()
        
        # Initialize HuggingFace embeddings with explicit cleanup
        self.embeddings = None
        self._init_embeddings()
        
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
        self.doc_queue: asyncio.Queue[Optional[Document]] = asyncio.Queue(maxsize=1000)
        
        # Track running tasks
        self.tasks: Set[asyncio.Task] = set()
        self.embedding_workers: List[asyncio.Task] = []
        
        # For graceful shutdown
        self.shutdown_requested = False
        self._is_closed = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Load or create vector store
        self._load_vectorstore()
    
    def _init_embeddings(self):
        """Initialize HuggingFace embeddings with proper cleanup"""
        if hasattr(self, 'embeddings') and self.embeddings is not None:
            self._cleanup_embeddings()
            
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _cleanup_embeddings(self):
        """Clean up HuggingFace embeddings resources"""
        if hasattr(self, 'embeddings') and self.embeddings is not None:
            if hasattr(self.embeddings, 'client'):
                try:
                    import torch
                    if hasattr(self.embeddings.client, 'to'):
                        self.embeddings.client.to('cpu')
                    if hasattr(self.embeddings.client, 'cpu'):
                        self.embeddings.client.cpu()
                    if hasattr(self.embeddings.client, 'to'):
                        del self.embeddings.client
                except Exception as e:
                    print(f"Warning: Error cleaning up embeddings: {e}")
            try:
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning: Error during GPU cleanup: {e}")
    
    # [Rest of the class methods remain exactly the same as in kb.py]
    # ...

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
        
    async def _crawl(self):
        """Main crawling loop"""
        connector = aiohttp.TCPConnector(limit=CONFIG["max_concurrent_requests"])
        async with aiohttp.ClientSession(connector=connector) as session:
            session.headers.update({"User-Agent": CONFIG["user_agent"]})
            
            last_save = 0
            save_interval = CONFIG["save_interval"]
            
            while self.pages_to_visit and not self.shutdown_requested:
                try:
                    if self.max_pages > 0 and len(self.visited_urls) >= self.max_pages:
                        print(f"Reached maximum page limit of {self.max_pages} pages.")
                        break
                        
                    # Get next URL to visit
                    try:
                        url = self.pages_to_visit.popleft()
                    except IndexError:
                        break
                        
                    if url in self.visited_urls:
                        continue
                        
                    print(f"Processing: {url}")
                    self.visited_urls.add(url)
                    
                    # Fetch the page
                    result = await self.fetch_page(session, url)
                    
                    # Process the page if successful
                    if result and result.content:
                        await self.process_document(result.url, result.content)
                        
                        # Add new links to the queue
                        if result.links:
                            for link in result.links:
                                if (link not in self.visited_urls and 
                                    link not in self.pages_to_visit and
                                    link != url):
                                    self.pages_to_visit.append(link)
                    
                    # Save periodically, but not too often
                    if (self.batches_processed > 0 and 
                        self.batches_processed - last_save >= save_interval and
                        self.vectorstore is not None):
                        print(f"Saving progress... (processed {self.batches_processed} batches)")
                        try:
                            # Create a temporary directory for atomic save
                            temp_dir = Path(f"{self.persist_dir}_temp")
                            if temp_dir.exists():
                                import shutil
                                shutil.rmtree(temp_dir, ignore_errors=True)
                                
                            # Save to temp directory first
                            self.vectorstore.save_local(folder_path=str(temp_dir), index_name="index")
                            
                            # Then move to final location atomically
                            if Path(self.persist_dir).exists():
                                import shutil
                                shutil.rmtree(self.persist_dir)
                            temp_dir.rename(self.persist_dir)
                            
                            last_save = self.batches_processed
                            print(f"Progress saved successfully at batch {last_save}")
                            
                        except Exception as save_error:
                            print(f"Error saving progress: {save_error}")
                            # Continue crawling even if save fails
                            
                except Exception as e:
                    print(f"Error processing URL {url}: {e}")
                    continue
    
    async def process_document(self, url: str, content: str) -> None:
        """Process a single document and add it to the queue"""
        if not content or not isinstance(content, str) or len(content.strip()) < 10:
            print(f"Skipping empty or invalid content for {url}")
            return
            
        try:
            print(f"Processing document from {url} (length: {len(content)} chars)")
            
            # Clean up content
            content = ' '.join(content.split())  # Normalize whitespace
            
            # Split the content into chunks
            try:
                chunks = self.text_splitter.split_text(content)
                if not chunks:
                    print(f"No chunks created for {url}")
                    return
                    
                print(f"Split into {len(chunks)} chunks for {url}")
                
                # Create documents for each chunk
                for i, chunk in enumerate(chunks):
                    if not chunk or len(chunk.strip()) < 10:  # Skip very small chunks
                        print(f"Skipping small chunk {i+1}/{len(chunks)} for {url}")
                        continue
                        
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": url,
                            "chunk": i,
                            "total_chunks": len(chunks),
                            "timestamp": datetime.now().isoformat(),
                            "url": url
                        }
                    )
                    await self.doc_queue.put(doc)
                    self.documents_processed += 1
                    
                    if self.documents_processed % 10 == 0:
                        print(f"Processed {self.documents_processed} chunks total...")
                        
            except Exception as split_error:
                print(f"Error splitting content for {url}: {split_error}")
                # Try to add the whole content as one document if splitting fails
                print("Attempting to add as a single document...")
                doc = Document(
                    page_content=content[:10000],  # Limit size
                    metadata={
                        "source": url,
                        "chunk": 0,
                        "total_chunks": 1,
                        "timestamp": datetime.now().isoformat(),
                        "url": url,
                        "error": str(split_error)
                    }
                )
                await self.doc_queue.put(doc)
                self.documents_processed += 1
                    
        except Exception as e:
            print(f"Error processing {url}: {e}")
            import traceback
            traceback.print_exc()
    
    async def _embedding_worker(self):
        """Worker that processes documents and generates embeddings"""
        try:
            while not self.shutdown_requested:
                try:
                    # Add a small delay to prevent busy waiting
                    try:
                        doc = await asyncio.wait_for(self.doc_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if self.shutdown_requested:
                            break
                        continue
                        
                    if doc is None:  # Sentinel value to stop the worker
                        break
                        
                    try:
                        # Add the document to the vector store
                        async with self._save_lock:
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.vectorstore.add_documents([doc])
                            )
                        
                        self.batches_processed += 1
                        
                        if self.batches_processed % CONFIG["save_interval"] == 0:
                            print(f"Processed {self.batches_processed} batches of documents")
                            
                    except Exception as e:
                        print(f"Error processing document: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        self.doc_queue.task_done()
                        
                except Exception as e:
                    print(f"Error in embedding worker: {e}")
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(1)  # Prevent tight loop on errors
                    
        except asyncio.CancelledError:
            print("Embedding worker cancelled")
            raise
        except Exception as e:
            print(f"Fatal error in embedding worker: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Embedding worker shutting down")
                
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> CrawlResult:
        """Fetch a single page and extract content and links"""
        try:
            print(f"Fetching URL: {url}")
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=CONFIG["request_timeout"])) as response:
                if response.status != 200:
                    error_msg = f"HTTP {response.status}"
                    print(f"Error fetching {url}: {error_msg}")
                    return CrawlResult(url, None, [], error_msg)
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract main content (trying multiple selectors)
                selectors = [
                    'main',
                    'article',
                    'div.content',
                    'div.main-content',
                    'div#content',
                    'div#main',
                    'div.page-content',
                    'div.entry-content',
                    'div.post-content',
                    'div#primary',
                    'div#main-content',
                    'div.region-content',
                    'div#content-area',
                    'div#main-content-area',
                    'div#main-content-wrapper',
                    'div#main-wrapper',
                    'div#content-wrapper',
                    'div#main-content-area',
                    'div#main-body',
                    'div.main',
                    'div.content-area',
                    'div.site-content',
                    'div#content-area',
                    'div#main-content-area',
                    'div#main-content-wrapper',
                    'div#main-wrapper',
                    'div#content-wrapper',
                    'div#main-content-area',
                    'div#main-body',
                    'div.main',
                    'div.content-area',
                    'div.site-content',
                    'body'
                ]
                
                main_content = None
                for selector in selectors:
                    main_content = soup.select_one(selector)
                    if main_content is not None:
                        break
                        
                if main_content is None:
                    print(f"Warning: Could not find main content in {url}")
                    # Try to use the whole body if no specific content found
                    main_content = soup.body or soup
                
                content = ' '.join(main_content.stripped_strings) if main_content else None
                
                if not content or len(content.strip()) < 100:  # If content seems too small
                    print(f"Warning: Very small or empty content for {url}")
                
                # Extract links
                links = []
                seen_links = set()
                for link in soup.find_all('a', href=True):
                    try:
                        href = link['href'].strip()
                        if not href or href.startswith('#'):
                            continue
                            
                        absolute_url = urljoin(url, href)
                        absolute_url = absolute_url.split('#')[0]  # Remove fragments
                        
                        if (self.is_valid_url(absolute_url) and 
                            absolute_url not in self.visited_urls and 
                            absolute_url not in seen_links):
                            links.append(absolute_url)
                            seen_links.add(absolute_url)
                    except Exception as e:
                        print(f"Error processing link {link.get('href', '')}: {e}")
                
                print(f"Fetched {url} - Content length: {len(content) if content else 0} chars, Links found: {len(links)}")
                return CrawlResult(url, content, links)
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error fetching {url}: {error_msg}")
            return CrawlResult(url, None, [], error_msg)

    async def start(self):
        """Start the scraping process"""
        if self.shutdown_requested or self._is_closed:
            raise RuntimeError("Scraper has been shut down and cannot be restarted")
            
        print(f"Starting KB scraper with base URL: {self.base_url}")
        print(f"Maximum pages to scrape: {self.max_pages if self.max_pages > 0 else 'unlimited'}")
        
        try:
            # Start the embedding workers
            for _ in range(CONFIG["max_embedding_workers"]):
                worker = asyncio.create_task(self._embedding_worker())
                self.embedding_workers.append(worker)
                self.tasks.add(worker)
                worker.add_done_callback(self.tasks.discard)
            
            # Start crawling
            await self._crawl()
            
            # Signal workers to finish
            for _ in range(len(self.embedding_workers)):
                await self.doc_queue.put(None)
            
            # Wait for queue to be processed
            print("Waiting for queue to empty...")
            await self.doc_queue.join()
            
            # Cancel any remaining tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
        except asyncio.CancelledError:
            print("\nCancellation requested. Finishing up...")
            self.shutdown_requested = True
            raise
        except Exception as e:
            print(f"\nError in scraper: {e}")
            import traceback
            traceback.print_exc()
            self.shutdown_requested = True
            raise
            
    async def close(self):
        """Save the vector store and clean up"""
        if self._is_closed:
            return
            
        self.shutdown_requested = True
        
        try:
            # Cancel all running tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Save the vector store if it exists
            if self.vectorstore is not None:
                print("\nSaving final vector store state...")
                try:
                    # Create a temporary directory for atomic save
                    temp_dir = Path(f"{self.persist_dir}_temp")
                    if temp_dir.exists():
                        import shutil
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    
                    # Save to temp directory first
                    async with self._save_lock:
                        self.vectorstore.save_local(folder_path=str(temp_dir), index_name="index")
                    
                    # Then move to final location atomically
                    if Path(self.persist_dir).exists():
                        import shutil
                        shutil.rmtree(self.persist_dir)
                    temp_dir.rename(self.persist_dir)
                    
                    print(f"Final state saved successfully with {self.vectorstore.index.ntotal} vectors")
                    
                except Exception as save_error:
                    print(f"Error during final save: {save_error}")
                    # Try one more time with direct save
                    try:
                        self.vectorstore.save_local(folder_path=str(self.persist_dir), index_name="index")
                        print("Fallback save completed")
                    except Exception as e:
                        print(f"Fallback save also failed: {e}")
            
            # Clean up embeddings
            self._cleanup_embeddings()
            
            # Clear queues and collections
            while not self.doc_queue.empty():
                try:
                    self.doc_queue.get_nowait()
                    self.doc_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            self.visited_urls.clear()
            self.pages_to_visit.clear()
            self.tasks.clear()
            self.embedding_workers.clear()
            
            if hasattr(self, 'vectorstore') and self.vectorstore is not None:
                try:
                    del self.vectorstore
                except Exception as e:
                    print(f"Error cleaning up vector store: {e}")
                self.vectorstore = None
            
            print("Cleanup complete.")
            
        finally:
            self._is_closed = True
    
    def __del__(self):
        """Ensure resources are cleaned up when the object is garbage collected"""
        if not self._is_closed and not self.shutdown_requested:
            print("\nWarning: KBScraper was not properly closed. Cleaning up...")
            try:
                # Try to run the cleanup synchronously
                if hasattr(self, '_cleanup_embeddings'):
                    self._cleanup_embeddings()
                
                # Clear any remaining references
                if hasattr(self, 'vectorstore') and self.vectorstore is not None:
                    try:
                        del self.vectorstore
                    except:
                        pass
                    self.vectorstore = None
                
                # Clear collections
                if hasattr(self, 'visited_urls'):
                    self.visited_urls.clear()
                if hasattr(self, 'pages_to_visit'):
                    self.pages_to_visit.clear()
                if hasattr(self, 'tasks'):
                    self.tasks.clear()
                if hasattr(self, 'embedding_workers'):
                    self.embedding_workers.clear()
                    
            except Exception as e:
                print(f"Error during final cleanup: {e}")
            finally:
                self._is_closed = True
                self.shutdown_requested = True


def install_required_packages():
    """Ensure all required packages are installed"""
    import sys
    import subprocess
    import importlib.util
    
    required_packages = [
        ('aiohttp', 'aiohttp'),
        ('bs4', 'beautifulsoup4'),
        ('langchain_huggingface', 'langchain-huggingface'),
        ('langchain_community', 'langchain-community'),
        ('langchain_core', 'langchain-core'),
        ('sentence_transformers', 'sentence-transformers'),
        ('faiss', 'faiss-cpu'),  # or faiss-gpu if you have CUDA
    ]
    
    for import_name, package_name in required_packages:
        if importlib.util.find_spec(import_name) is None:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


async def main():
    """Main function to run the KB scraper"""
    try:
        # Ensure required packages are installed
        install_required_packages()
        
        # You can customize these parameters as needed
        base_url = "https://kb.wisc.edu/"
        max_pages = 1000000  # Start with a small number for testing
        persist_dir = "faiss_index"
        
        print("\n" + "="*50)
        print("KB Scraper - Starting up")
        print("="*50)
        
        # Create the scraper instance
        print(f"Initializing scraper for {base_url}")
        print(f"Persisting data to: {persist_dir}")
        print("-" * 50)
        
        scraper = KBScraper(
            base_url=base_url,
            max_pages=max_pages,
            persist_dir=persist_dir
        )
        
        try:
            # Start the scraping process
            print("\nStarting scraping process...")
            print("Press Ctrl+C to stop gracefully\n")
            await scraper.start()
            
        except Exception as e:
            print(f"\nError during scraping: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Ensure we save the final state
            print("\nShutting down...")
            await scraper.close()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nKB Scraper has finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Exiting...")
