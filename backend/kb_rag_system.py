import os
import re
import time
import signal
import asyncio
import aiohttp
import uuid
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Deque, Tuple, NamedTuple
from urllib.parse import urlparse, urljoin, unquote
from collections import deque, defaultdict
from bs4 import BeautifulSoup, Comment
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from backend.pinecone_service import PineconeService
from backend.utils.document_processor import DocumentProcessor, SUPPORTED_DOCUMENT_EXTENSIONS

# Import OpenAI at the top level
try:
    from openai import AsyncOpenAI
except ImportError:
    print("OpenAI package not installed. Install with: pip install openai")

# Set up logger
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definitions
class CrawlResult(NamedTuple):
    """Result of a page crawl operation."""
    url: str
    content: Optional[str]
    links: List[str]
    error: Optional[str] = None

# Configuration
CONFIG = {
    "max_concurrent_requests": 5,  # Number of concurrent HTTP requests
    "max_embedding_workers": 3,     # Number of concurrent embedding workers
    "request_timeout": 10,          # Request timeout in seconds
    "batch_size": 10,               # Number of documents to process in a batch
    "save_interval": 5,             # Save vector store every N batches
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Load environment variables from project root once at import
try:
    _PROJECT_ENV = (Path(__file__).resolve().parents[1] / ".env")
    load_dotenv(dotenv_path=_PROJECT_ENV, override=False)
except Exception:
    # Fall back to default lookup if specific path fails
    try:
        load_dotenv()
    except Exception:
        pass

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class KBScraper:
    def __init__(self, max_pages: int = 100000, user_id: str = "default", index_name: Optional[str] = None):
        """
        Initialize the KBScraper with Pinecone integration.
        
        Args:
            max_pages: Maximum number of pages to scrape
            user_id: Unique identifier for the user (used for namespacing in Pinecone)
            index_name: Optional name for the Pinecone index (defaults to user-specific name)
        """
        # Initialize attributes that are used in __del__ first
        self._is_closed = False
        self.shutdown_requested = False
        self.tasks = set()
        self.embedding_workers = []
        self.base_domain = None  # Will be set when process_website is called
        
        try:
            self.max_pages = max_pages
            self.user_id = user_id
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            # Cache Gemini credentials and model once
            self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            self.gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
            self.gemini_model = None
            self._genai = None
            if self.gemini_api_key:
                try:
                    import google.generativeai as genai  # type: ignore
                    genai.configure(api_key=self.gemini_api_key)
                    self._genai = genai
                    self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
                except Exception as _e:
                    logger.error(f"Failed to initialize Gemini model: {_e}")
            
            # Initialize Pinecone service and embeddings with the specified index name
            self.pinecone = PineconeService(user_id=user_id, index_name=index_name)
            self.embeddings = None
            self._init_embeddings()
            
            # Initialize document processor for handling PDFs, DOCs, etc.
            self.document_processor = DocumentProcessor(user_id=user_id, scraper=self)
            
            # No need for local vector store with Pinecone
            self.persist_dir = None
            
            # Queue for passing documents to embedding workers
            self.doc_queue: asyncio.Queue[Optional[Document]] = asyncio.Queue(maxsize=1000)
            
            # Initialize other attributes
            self.visited_urls = set()
            self.document_urls = set()  # Track document URLs separately
            self.pages_to_visit = deque()  # Use deque for efficient FIFO operations
            self.batches_processed = 0  # Track number of processed batches
            self.documents_processed = 0  # Track number of processed documents
            
            # Set up signal handlers
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
            
        except Exception as e:
            # If initialization fails, ensure resources are cleaned up
            self._is_closed = True
            raise
        
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
                    logger.warning(f"Error cleaning up embeddings: {e}")
            try:
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error during GPU cleanup: {e}")
    
    async def _shutdown_internal(self):
        """Internal method to handle the actual shutdown process"""
        if self._is_closed:
            return
            
        self.shutdown_requested = True
        
        try:
            # Cancel all running tasks with a timeout
            tasks_to_cancel = [t for t in self.tasks if not t.done()]
            if tasks_to_cancel:
                print(f"Waiting for {len(tasks_to_cancel)} tasks to complete...")
                for task in tasks_to_cancel:
                    task.cancel()
                
                # Use wait_for with a timeout to avoid hanging
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=5.0  # 5 second timeout
                    )
                except asyncio.TimeoutError:
                    print("Timeout waiting for tasks to cancel, forcing shutdown...")
            
            # Stop all embedding workers with timeout
            if hasattr(self, 'embedding_workers') and self.embedding_workers:
                print("Stopping embedding workers...")
                # Signal workers to stop by sending None
                for _ in self.embedding_workers:
                    try:
                        await asyncio.wait_for(self.doc_queue.put(None), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
                
                # Wait for workers to complete with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.embedding_workers, return_exceptions=True),
                        timeout=5.0  # 5 second timeout
                    )
                except asyncio.TimeoutError:
                    print("Timeout waiting for embedding workers, forcing shutdown...")
            
            # Clean up resources
            if hasattr(self, 'session') and not self.session.closed:
                await self.session.close()
                
            # Clean up embeddings
            self._cleanup_embeddings()
            
            print("Cleanup complete.")
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._is_closed = True
            print("Shutdown complete.")
            # Ensure we don't hang by explicitly exiting any running event loops
            # This is a last resort to prevent hanging
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.stop()
            except RuntimeError:
                pass
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("\nShutdown requested. Finishing current tasks...")
        self.shutdown_requested = True
        
        # Run the shutdown in the event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(self._shutdown_internal())
        else:
            loop.run_until_complete(self._shutdown_internal())
    
    def is_valid_url(self, url: str) -> bool:
        """
        Check if URL is valid, has a valid format, and belongs to the same domain.
        
        Args:
            url: The URL to validate
            
        Returns:
            bool: True if URL is valid and belongs to the same domain, False otherwise
        """
        try:
            if not url:
                return False
                
            parsed_url = urlparse(url)
            
            # Basic URL validation
            if not all([parsed_url.scheme, parsed_url.netloc]):
                return False
                
            # Skip non-HTTP/HTTPS URLs
            if parsed_url.scheme not in ['http', 'https']:
                return False
                
            # Check if this is a document URL
            path = parsed_url.path.lower()
            is_document = any(path.endswith(ext) for ext in SUPPORTED_DOCUMENT_EXTENSIONS.keys())
                
            # If base_domain is not set yet, this is the first URL being processed
            if self.base_domain is None:
                self.base_domain = parsed_url.netloc
                return True
                
            # Check if the URL belongs to the same domain (including subdomains)
            url_domain = parsed_url.netloc
            same_domain = (url_domain == self.base_domain or 
                          url_domain.endswith('.' + self.base_domain) or
                          ('.' + url_domain) in ('.' + self.base_domain))
                          
            return same_domain
                    
        except Exception as e:
            print(f"Error validating URL {url}: {e}")
            return False
    
    async def _load_vectorstore(self):
        """No-op for Pinecone as it handles persistence automatically"""
        pass
    
    async def _save_vectorstore(self):
        """No-op for Pinecone as it handles persistence automatically"""
        pass
    
    async def _crawl(self):
        """Main crawling loop"""
        # Initialize rate limiting and save tracking
        last_request_time = {}
        min_delay = 1.0  # Minimum delay between requests to the same domain in seconds
        last_save = time.time()
        save_interval = 300  # Save progress every 5 minutes
        
        # Add timeout for empty queue
        empty_queue_start = None
        empty_queue_timeout = 10.0  # Terminate after 10 seconds of empty queue
        
        connector = aiohttp.TCPConnector(limit=CONFIG["max_concurrent_requests"])
        async with aiohttp.ClientSession(connector=connector) as session:
            session.headers.update({"User-Agent": CONFIG["user_agent"]})
            
            while not self.shutdown_requested:
                # Check if queue is empty
                if not self.pages_to_visit:
                    if empty_queue_start is None:
                        empty_queue_start = time.time()
                        logger.info("Queue is empty, waiting for new URLs...")
                    elif time.time() - empty_queue_start > empty_queue_timeout:
                        logger.info(f"Queue has been empty for {empty_queue_timeout} seconds, terminating crawl")
                        break
                    await asyncio.sleep(1)
                    continue
                else:
                    empty_queue_start = None  # Reset empty queue timer
                
                url = None
                try:
                    url = self.pages_to_visit.popleft()
                    if url in self.visited_urls:
                        continue
                        
                    # Rate limiting by domain
                    domain = urlparse(url).netloc
                    current_time = time.time()
                    if domain in last_request_time:
                        time_since_last = current_time - last_request_time[domain]
                        if time_since_last < min_delay:
                            await asyncio.sleep(min_delay - time_since_last)
                    last_request_time[domain] = time.time()
                    
                    self.visited_urls.add(url)
                    if len(self.visited_urls) >= self.max_pages:
                        logger.info(f"Reached maximum page limit of {self.max_pages}")
                        break
                        
                    logger.info(f"Processing: {url}")
                    
                    try:
                        result = await asyncio.wait_for(
                            self.fetch_page(session, url),
                            timeout=60  # 1 minute timeout per page
                        )
                        
                        if result and result.error:
                            logger.warning(f"Error fetching {url}: {result.error}")
                            continue
                            
                        if result and result.content and len(result.content) > 100:  # Increased minimum content length
                            await self.process_document(result.url, result.content)
                            
                        if result and result.links:
                            new_links = 0
                            for link in result.links:
                                if (link not in self.visited_urls and 
                                    link not in self.pages_to_visit and
                                    link != url and
                                    self.is_valid_url(link)):
                                    self.pages_to_visit.append(link)
                                    new_links += 1
                            
                            logger.info(f"  + Found {len(result.links)} links, {new_links} new")
                    
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout processing {url}, skipping...")
                        continue
                        
                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}", exc_info=True)
                        try:
                            doc = Document(
                                page_content="",
                                metadata={
                                    "source": url,
                                    "chunk": 0,
                                    "total_chunks": 1,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "url": url,
                                    "error": f"Error processing URL: {str(e)[:200]}",
                                    "content_length": 0
                                }
                            )
                            await asyncio.wait_for(
                                self.doc_queue.put(doc),
                                timeout=10
                            )
                            self.documents_processed += 1
                        except Exception as inner_e:
                            logger.error(f"Failed to add document to queue: {inner_e}")
                        continue
                    
                    # Log progress periodically
                    if len(self.visited_urls) % 5 == 0:
                        logger.info(f"Status: {len(self.visited_urls)} pages, {len(self.pages_to_visit)} in queue, {self.documents_processed} documents processed")
                        
                        current_time = time.time()
                        if (current_time - last_save >= save_interval and 
                            self.documents_processed > 0):
                            logger.info(f"Progress: {self.documents_processed} documents processed, {len(self.visited_urls)} pages visited")
                            last_save = current_time
                
                except Exception as e:
                    if url:
                        logger.error(f"Unexpected error processing {url}: {e}", exc_info=True)
                    else:
                        logger.error(f"Unexpected error in crawl loop: {e}", exc_info=True)
                    await asyncio.sleep(1)  # Prevent tight loop on errors
            
            logger.info("Crawl process completed")
            # Signal completion
            if hasattr(self, '_crawl_task'):
                self._crawl_task = None

    async def process_document(self, url: str, content: str) -> None:
        """
        Process a document and add it to the Pinecone vector store.
        
        Args:
            url: The URL where the document was found
            content: The document content to process
        """
        try:
            if not content or len(content.strip()) < 50:  # Skip very short documents
                logger.warning(f"Skipping document from {url} - content too short")
                return
                
            # Create document metadata
            metadata = {
                "source": url,
                "url": url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content_length": len(content),
                "type": "webpage"
            }
            
            try:
                # Add the document to Pinecone
                self.pinecone.upsert_document(content, metadata)
                self.documents_processed += 1
                logger.info(f"Processed document from {url} - {len(content)} characters")
                
            except Exception as e:
                logger.error(f"Error adding document to vector store: {e}", exc_info=True)
                # Create an error document for tracking
                error_doc = Document(
                    page_content="",
                    metadata={
                        **metadata,
                        "error": f"Error processing document: {str(e)[:200]}",
                        "processed": False
                    }
                )
                await self.doc_queue.put(error_doc)
                
        except Exception as e:
            logger.error(f"Unexpected error in process_document for {url}: {e}", exc_info=True)

    async def _embedding_worker(self):
        """Worker that processes documents and generates embeddings"""
        try:
            while not self.shutdown_requested:
                try:
                    # Add a small delay to prevent busy waiting
                    try:
                        doc = await asyncio.wait_for(self.doc_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
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
                            logger.info(f"Processed {self.batches_processed} batches of documents")
                            
                    except Exception as e:
                        logger.error(f"Error processing document: {e}", exc_info=True)
                        continue
                    finally:
                        self.doc_queue.task_done()
                        
                except Exception as e:
                    logger.error(f"Error in embedding worker: {e}", exc_info=True)
                    await asyncio.sleep(1)  # Prevent tight loop on errors
                    
        except asyncio.CancelledError:
            logger.info("Embedding worker cancelled")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error in embedding worker: {e}", exc_info=True)
            raise
            
        finally:
            logger.info("Embedding worker shutting down")
                
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[CrawlResult]:
        """Fetch a single page and extract content and links"""
        max_retries = 3
        retry_delay = 1  # seconds
        
        # Check if this is a document URL
        if self.document_processor.is_document_url(url):
            # If we haven't processed this document yet
            if url not in self.document_urls:
                # Queue the document for processing
                self.document_urls.add(url)
                await self.document_processor.queue_document(url, self.base_domain)
                logger.info(f"Queued document URL for processing: {url}")
            
            # Return an empty result since we're handling this separately
            return CrawlResult(url, None, [], None)
        
        for attempt in range(max_retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        if response.status in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                            # Exponential backoff for server errors
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                            continue
                        return CrawlResult(url, None, [], f"HTTP {response.status}")
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Common content selectors - ordered by likelihood of containing main content
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
                        'div#main-body',
                        'div.main',
                        'div.content-area',
                        'div.site-content',
                        'body'
                    ]
                    
                    # Try to find main content using selectors
                    main_content = None
                    for selector in selectors:
                        main_content = soup.select_one(selector)
                        if main_content is not None:
                            break
                            
                    if main_content is None:
                        print(f"Warning: Could not find main content in {url}")
                        # Fall back to body if no specific content found
                        main_content = soup.body or soup
                    
                    # Clean up the content
                    for element in main_content.select('script, style, nav, footer, header, aside, form, iframe'):
                        element.decompose()
                    
                    content = ' '.join(main_content.stripped_strings) if main_content else None
                    
                    if not content or len(content.strip()) < 100:  # If content seems too small
                        print(f"Warning: Very small or empty content for {url}")
                    
                    # Extract links with deduplication
                    links = []
                    seen_links = set()
                    document_links = []
                    
                    for link in soup.find_all('a', href=True):
                        try:
                            href = link['href'].strip()
                            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                                continue
                                
                            absolute_url = urljoin(url, href)
                            parsed = urlparse(absolute_url)
                            
                            # Clean and normalize the URL
                            clean_url = parsed._replace(
                                fragment='',
                                query='',
                                params=''
                            ).geturl()
                            
                            # Check if this is a document URL
                            is_document = self.document_processor.is_document_url(clean_url)
                            
                            if (self.is_valid_url(clean_url) and 
                                clean_url not in self.visited_urls and 
                                clean_url not in seen_links):
                                
                                if is_document:
                                    if clean_url not in self.document_urls:
                                        document_links.append(clean_url)
                                else:
                                    links.append(clean_url)
                                
                                seen_links.add(clean_url)
                        except Exception as e:
                            print(f"Error processing link {link.get('href', '')}: {e}")
                    
                    # Queue document links for processing
                    for doc_url in document_links:
                        self.document_urls.add(doc_url)
                        await self.document_processor.queue_document(doc_url, self.base_domain)
                    
                    print(f"Fetched {url} - Content length: {len(content) if content else 0} chars, Links found: {len(links)}, Documents found: {len(document_links)}")
                    return CrawlResult(url, content, links)
                
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if attempt == max_retries - 1:
                    return CrawlResult(url, None, [], f"Request failed after {max_retries} attempts: {str(e)}")
                await asyncio.sleep(retry_delay * (2 ** attempt))
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error fetching {url}: {error_msg}")
                if attempt == max_retries - 1:
                    return CrawlResult(url, None, [], f"Failed after {max_retries} attempts: {error_msg}")
                await asyncio.sleep(retry_delay * (2 ** attempt))
        
        return CrawlResult(url, None, [], "Max retries exceeded")

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
        """Close the scraper and clean up resources"""
        if self._is_closed:
            return
        
        self._is_closed = True
        self.shutdown_requested = True
        
        logger.info("Closing KB scraper...")
        
        try:
            # Cancel any running crawl task
            if hasattr(self, '_crawl_task') and not self._crawl_task.done():
                self._crawl_task.cancel()
                try:
                    await self._crawl_task
                except asyncio.CancelledError:
                    pass
            
            # Shut down the document processor
            if hasattr(self, 'document_processor'):
                await self.document_processor.shutdown()
                logger.info("Document processor shut down")
            
            # Clear any remaining items in the queue
            await self._clear_queues()
            
            # Cancel any remaining tasks
            for task in list(self.tasks):
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Clean up embeddings
            self._cleanup_embeddings()
            
            logger.info("KB scraper closed successfully")
        except Exception as e:
            logger.error(f"Error closing KB scraper: {e}")
            # Continue with shutdown even if there are errors
    
    async def _clear_queues(self):
        """Clear all queues and collections"""
        # Clear the document queue
        try:
            while not self.doc_queue.empty():
                try:
                    self.doc_queue.get_nowait()
                    self.doc_queue.task_done()
                except asyncio.QueueEmpty:
                    break
        except Exception as e:
            logger.error(f"Error clearing document queue: {e}")
        
        # Clear collections
        self.visited_urls.clear()
        self.document_urls.clear()
        self.pages_to_visit.clear()
        
        logger.info("All queues and collections cleared")
    
    async def process_website(self, url: str) -> Dict[str, Any]:
        """Process a website and add it to the knowledge base
        
        Args:
            url: The URL of the website to process (this will be used as the base for crawling)
            
        Returns:
            Dict with status and message
        """
        try:
            # Validate the initial URL
            if not self.is_valid_url(url):
                return {
                    "status": "error",
                    "message": f"Invalid or unsupported URL: {url}"
                }
            
            # Check if a crawl is already in progress
            crawl_in_progress = hasattr(self, '_crawl_task') and not self._crawl_task.done()
                
            # Add the URL to the pages to visit if it's not already there
            if url not in self.pages_to_visit and url not in self.visited_urls:
                self.pages_to_visit.append(url)
                print(f"Added initial URL to crawl: {url} (base domain: {self.base_domain})")
            elif url in self.visited_urls:
                return {
                    "status": "info",
                    "message": f"URL {url} has already been processed"
                }
            
            # Start the document processor if not already running
            await self.document_processor.start_processing()
            
            # Start the crawling process if not already running
            if not crawl_in_progress:
                print(f"Starting new crawl process for URL: {url}")
                self._crawl_task = asyncio.create_task(self._crawl())
                return {
                    "status": "success",
                    "message": f"Website added to processing queue and crawl started: {url}"
                }
            else:
                return {
                    "status": "success",
                    "message": f"Website added to processing queue (crawl already in progress): {url}. Current status: {len(self.visited_urls)} pages visited, {len(self.pages_to_visit)} in queue."
                }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to process website: {str(e)}"
            }
    
    async def process_pdf(self, file_path: str) -> Dict[str, str]:
        """Process a PDF file and add it to the knowledge base
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dict with status and message
        """
        try:
            from PyPDF2 import PdfReader
            
            # Read the PDF file
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                # Process the extracted text
                if not text.strip():
                    return {"status": "error", "message": "No text could be extracted from the PDF"}
                
                # Add the document to the knowledge base
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.process_document(file_path, text)
                )
                
                return {"status": "success", "message": "PDF processed successfully"}
                
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {"status": "error", "message": f"Failed to process PDF: {str(e)}"}
    
    async def query(self, query: str, k: int = 5, conversation_history: Optional[List[Dict[str, str]]] = None, context_documents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query the knowledge base using Gemini by default, with OpenAI as fallback.
        
        Args:
            query: The query string
            k: Number of results to return
            conversation_history: Optional list of previous messages in the conversation
            context_documents: Optional list of context documents to include
        
        Returns:
            Dict with status, answer, and optional error fields
        """
        import time
        start_time = time.time()
        
        try:
            # Query Pinecone for similar documents
            results = await self.pinecone.query(query, k=k)
            
            if not results:
                return {"status": "success", "answer": "The knowledge base is currently empty. Please add some documents first.", "sources": []}
            
            # Format context and sources
            context = "\n\n".join([r["text"] for r in results])
            sources = list(set([r["source"] for r in results if "source" in r]))
            
            # Add any additional context documents if provided
            if context_documents and len(context_documents) > 0:
                additional_context = "\n\n".join(context_documents)
                context = f"{context}\n\n{additional_context}"
            
            # Format conversation history if provided
            conversation_context = ""
            if conversation_history and len(conversation_history) > 0:
                conversation_context = "Previous conversation:\n"
                for msg in conversation_history:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    conversation_context += f"{role.capitalize()}: {content}\n"
                conversation_context += "\n"
            
            # Primary (and only): Gemini
            # Build the Gemini prompt once
            gemini_prompt = f"""You are a knowledgeable support agent working for the organization. 
            Consider the following passages now a part of your personal knowledge base.
            
            Your task has TWO steps:
            1. Identify which passages are most relevant to the user’s question.
            - Consider accuracy, specificity, and directness.
            
            2. Using the most relevant passages, write a clear, professional, and confident answer to the user’s question.
            However, if the answer is not directly in the context, infer as much as you can from the provided passages and search through your own general knowledge, 
            and offer the best possible solution
            - Mirror the language style of the passages where possible.
            - Be concise and direct: include only as much detail as needed to resolve the issue.
            {context}
            
            {conversation_context}
            
            Question: {query}
            
            As the frontline support agent, your goal is to resolve the user’s issue with clarity, kindness, and authority. 
            """
            
            print("[RAG] Trying Gemini as primary option...")
            if self.gemini_api_key and self.gemini_model is not None:
                try:
                    print(f"[RAG][LLM] Querying Gemini with question: {query}")
                    print(f"[RAG][LLM] Gemini model: {self.gemini_model_name}")
                    import asyncio as _asyncio
                    def _gen_content():
                        return self.gemini_model.generate_content(
                            gemini_prompt,
                            generation_config={
                                "temperature": 0.2,
                                "max_output_tokens": 4096,
                                "response_mime_type": "text/plain",
                            },
                        )
                    resp = await _asyncio.to_thread(_gen_content)
                    # Safely extract text
                    answer = ""
                    try:
                        candidates = getattr(resp, "candidates", None)
                        if candidates:
                            for c in candidates:
                                content = getattr(c, "content", None)
                                parts = getattr(content, "parts", []) if content else []
                                for p in parts:
                                    t = getattr(p, "text", None)
                                    if t:
                                        answer += t
                        # If still empty, check prompt_feedback for block reason
                        if not answer:
                            # Debug finish reasons when empty
                            try:
                                fins = [getattr(c, "finish_reason", None) for c in (candidates or [])]
                                print(f"[RAG][Gemini] finish_reasons: {fins}")
                            except Exception:
                                pass
                            # Last resort, try quick accessor
                            try:
                                quick = getattr(resp, "text", None)
                                if isinstance(quick, str) and quick.strip():
                                    answer = quick
                            except Exception:
                                pass
                            prompt_feedback = getattr(resp, "prompt_feedback", None)
                            if prompt_feedback and getattr(prompt_feedback, "block_reason", None):
                                answer = f"[Gemini blocked: {prompt_feedback.block_reason}]"
                    except Exception:
                        answer = ""
                    answer = (answer or "").strip()
                    if answer:
                        return {
                            "status": "success",
                            "answer": answer,
                            "sources": sources,
                            "model_used": f"gemini/{self.gemini_model_name}"
                        }
                except Exception as e:
                    logger.error(f"Gemini error: {e}")
                    return {"status": "error", "answer": f"Gemini error: {e}", "sources": sources}
            else:
                print("No LLM provider configured. Please set GEMINI_API_KEY/GOOGLE_API_KEY.")
                return {"status": "error", "answer": "No LLM provider configured. Please set GEMINI_API_KEY/GOOGLE_API_KEY.", "sources": sources}
        except Exception as e:
            error_time = time.time() - start_time
            print(f"\n=== Error in Query ===")
            print(f"Error: {str(e)}")
            print(f"Error occurred after {error_time:.2f}s")
            print("=======================\n")
            return {
                "status": "error",
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": []
            }
    
    def update_index_name(self, index_name: str):
        """Update the index name for the scraper"""
        if self.pinecone:
            self.pinecone = PineconeService(user_id=self.user_id, index_name=index_name)
            logger.info(f"Updated index name to: {index_name}")
    
    def get_index_name(self) -> str:
        """Get the current index name"""
        if self.pinecone:
            return self.pinecone.index_name
        return None

    async def list_available_models(self) -> List[str]:
        """Get a list of available Gemini models
        
        Returns:
            List of model names
        """
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("No GEMINI_API_KEY found in environment")
                return []
            
            genai.configure(api_key=api_key)
            models = await asyncio.to_thread(genai.list_models)
            return [m.name for m in models]
        except Exception as e:
            print(f"Error listing Gemini models: {e}")
            # Return empty list on error so we can fall back to default models
            return []

    def __del__(self):
        """Ensure resources are cleaned up when the object is garbage collected"""
        if not getattr(self, '_is_closed', True) and not getattr(self, 'shutdown_requested', False):
            print("\nWarning: KBScraper was not properly closed. Cleaning up...")
            try:
                # Don't use asyncio in __del__, it's not safe
                # Just do minimal cleanup
                self._is_closed = True
                self.shutdown_requested = True
                
                # Clean up embeddings directly
                if hasattr(self, '_cleanup_embeddings'):
                    self._cleanup_embeddings()
                
                # Clear any remaining references
                if hasattr(self, 'vectorstore') and self.vectorstore is not None:
                    try:
                        del self.vectorstore
                    except Exception:
                        pass
                    self.vectorstore = None
                
                # Clear collections
                for attr in ['visited_urls', 'pages_to_visit', 'tasks', 'embedding_workers']:
                    if hasattr(self, attr):
                        try:
                            collection = getattr(self, attr)
                            if hasattr(collection, 'clear'):
                                collection.clear()
                        except Exception:
                            pass
                
                print("Emergency cleanup complete")
                
            except Exception:
                pass


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
        start_url = input("Enter the URL to start scraping from: ").strip()
        max_pages = 1000000  # Maximum number of pages to scrape
        
        print("\n" + "="*50)
        print("KB Scraper - Starting up")
        print("="*50)
        
        # Create the scraper instance
        print(f"Initializing scraper")
        print("-" * 50)
        
        scraper = KBScraper(max_pages=max_pages)
        
        try:
            # Start the scraping process with the provided URL
            print(f"\nStarting scraping process from: {start_url}")
            print("Press Ctrl+C to stop gracefully\n")
            await scraper.process_website(start_url)
            
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
