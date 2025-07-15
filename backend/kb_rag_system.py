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
from pinecone_service import PineconeService

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

# Load environment variables
load_dotenv()

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class KBScraper:
    def __init__(self, max_pages: int = 100000, user_id: str = "default"):
        """
        Initialize the KBScraper with Pinecone integration.
        
        Args:
            max_pages: Maximum number of pages to scrape
            user_id: Unique identifier for the user (used for namespacing in Pinecone)
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
            self.gemini_model = None
            self.gemini_model_name = None
            
            # Initialize Pinecone service and embeddings
            self.pinecone = PineconeService(user_id=user_id)
            self.embeddings = None
            self._init_embeddings()
            
            # No need for local vector store with Pinecone
            self.persist_dir = None
            
            # Queue for passing documents to embedding workers
            self.doc_queue: asyncio.Queue[Optional[Document]] = asyncio.Queue(maxsize=1000)
            
            # Initialize other attributes
            self.visited_urls = set()
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
                
            # Skip non-HTTP/HTTPS URLs and file extensions we don't want to process
            if (parsed_url.scheme not in ['http', 'https'] or
                any(ext in url.lower() for ext in ['.pdf', '.jpg', '.jpeg', '.png', '.gif'])):
                return False
                
            # If base_domain is not set yet, this is the first URL being processed
            if self.base_domain is None:
                self.base_domain = parsed_url.netloc
                return True
                
            # Check if the URL belongs to the same domain (including subdomains)
            url_domain = parsed_url.netloc
            return (url_domain == self.base_domain or 
                   url_domain.endswith('.' + self.base_domain) or
                   ('.' + url_domain) in ('.' + self.base_domain))
                    
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
                            
                            if (self.is_valid_url(clean_url) and 
                                clean_url not in self.visited_urls and 
                                clean_url not in seen_links):
                                links.append(clean_url)
                                seen_links.add(clean_url)
                        except Exception as e:
                            print(f"Error processing link {link.get('href', '')}: {e}")
                    
                    print(f"Fetched {url} - Content length: {len(content) if content else 0} chars, Links found: {len(links)}")
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
        """Save the vector store and clean up"""
        if self._is_closed:
            return
            
        self.shutdown_requested = True
        
        try:
            # Cancel the crawl task if it exists
            if hasattr(self, '_crawl_task') and not self._crawl_task.done():
                print("Cancelling active crawl task...")
                self._crawl_task.cancel()
                try:
                    await asyncio.wait_for(self._crawl_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    print("Crawl task cancelled")
            
            # Cancel all running tasks with timeout
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete with timeout
            if self.tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.tasks, return_exceptions=True),
                        timeout=5.0  # 5 second timeout
                    )
                except asyncio.TimeoutError:
                    print("Timeout waiting for tasks to complete during close")
            
            # Save the vector store if it exists, but with a timeout
            if hasattr(self, 'vectorstore') and self.vectorstore is not None:
                print("\nSaving final vector store state...")
                try:
                    # Use a simpler save approach to avoid hanging
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            self.vectorstore.save_local,
                            folder_path=str(self.persist_dir),
                            index_name="index"
                        ),
                        timeout=30.0  # 30 second timeout for saving
                    )
                    print(f"Final state saved successfully")
                except (asyncio.TimeoutError, Exception) as e:
                    print(f"Error during vector store save: {e}")
            
            # Clean up embeddings
            self._cleanup_embeddings()
            
            # Clear queues and collections with timeout
            try:
                await asyncio.wait_for(self._clear_queues(), timeout=5.0)
            except asyncio.TimeoutError:
                print("Timeout clearing queues, forcing cleanup")
            
            # Clear collections
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
    
    async def _clear_queues(self):
        """Clear queues safely"""
        while not self.doc_queue.empty():
            try:
                self.doc_queue.get_nowait()
                self.doc_queue.task_done()
            except asyncio.QueueEmpty:
                break
    
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
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"
            
            # Process the text as a document
            doc = Document(
                page_content=text,
                metadata={"source": file_path}
            )
            
            # Add to vector store
            if not hasattr(self, 'vectorstore') or self.vectorstore is None:
                self._load_vectorstore()
                
            self.vectorstore.add_documents([doc])
            
            # Save the updated vector store
            self.vectorstore.save_local(folder_path=str(self.persist_dir), index_name="index")
            
            return {
                "status": "success",
                "message": f"Successfully processed PDF: {file_path}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to process PDF: {str(e)}"
            }
    
    async def list_available_models(self):
        """List all available models from the Gemini API"""
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        try:
            models = genai.list_models()
            return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    async def query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Query the knowledge base using OpenAI first, then fall back to Gemini if needed
        
        Args:
            query: The query string
            k: Number of results to return
            
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
            
            # First try OpenAI as the primary option
            print("Trying OpenAI as primary option...")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                try:
                    # Use the imported AsyncOpenAI
                    client = AsyncOpenAI(api_key=openai_api_key)
                    
                    # Format the prompt for OpenAI
                    openai_prompt = f"""You are a knowledgeable IT support assistant. Use the following context to answer the question.
            
                        Context:
                        {context}
                        
                        Question: {query}
                        
                        You are a knowledgeable, confident customer support assistant.

                        Speak with clarity and authority — your tone should instill trust.

                        Avoid vague or wishy-washy language. Do not use words like "maybe," "possibly," "I think," or "it appears." Always give the most direct, helpful answer possible.

                        The context below is your own knowledge: integrate it seamlessly into your answers.

                        If a question involves a tool or link, provide the direct URL.

                        Avoid acronyms and technical jargon unless absolutely necessary.

                        If the answer is in the context, respond with a clear and direct explanation based on that information.
                        If the answer is not in the context, use general knowledge and similar scenarios to offer the best possible solution — never say "I don't know."

                        If your answer is based on the provided context, include the most relevant source URL at the end of your reply."""
                    
                    # Try GPT-4o-mini first, fall back to gpt-3.5-turbo if needed
                    models_to_try = ["gpt-4o-mini", "gpt-3.5-turbo"]
                    last_openai_error = None
                    
                    for model in models_to_try:
                        try:
                            print(f"Trying OpenAI model: {model}")
                            response = await client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant that provides accurate and concise answers."},
                                    {"role": "user", "content": openai_prompt}
                                ],
                                max_tokens=1000,
                                temperature=0.7
                            )
                            
                            answer = response.choices[0].message.content.strip()
                            
                            return {
                                "status": "success",
                                "answer": answer,
                                "sources": sources,
                                "model_used": f"openai/{model}"
                            }
                            
                        except Exception as e:
                            last_openai_error = e
                            print(f"Error with OpenAI model {model}: {e}")
                            continue
                            
                    # If we get here, all OpenAI models failed
                    print(f"All OpenAI models failed. Last error: {last_openai_error}")
                    print("Falling back to Gemini...")
                    
                except Exception as e:
                    print(f"OpenAI API initialization error: {e}")
                    print("Falling back to Gemini...")
            else:
                print("No OpenAI API key found, falling back to Gemini...")
            
            # If OpenAI failed or no API key, fall back to Gemini
            # 4. Initialize Gemini model if not already done
            if not hasattr(self, 'gemini_model'):
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY environment variable is not set")
                
                genai.configure(api_key=api_key)
                
                # Try to find the best available model
                try:
                    # Preferred models in order of preference
                    preferred_models = [
                        'gemini-1.5-flash-latest',
                        'gemini-1.5-pro-latest',
                        'gemini-pro',
                    ]
                    
                    # Get available models
                    available_models = [m.name.split('/')[-1] for m in genai.list_models()]
                    
                    # Find the first preferred model that's available
                    model_name = next((m for m in preferred_models if m in available_models), None)
                    
                    if not model_name and available_models:
                        # If no preferred model is found, use the first available model that supports text
                        for m in available_models:
                            if 'gemini' in m.lower() and 'vision' not in m.lower():
                                model_name = m
                                break
                    
                    if not model_name and available_models:
                        # Last resort: use any available model
                        model_name = available_models[0]
                    
                    if not model_name:
                        raise RuntimeError("No suitable Gemini model found")
                        
                    print(f"Using Gemini model: {model_name}")
                    self.gemini_model = genai.GenerativeModel(model_name)
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")
            
            # 5. Create prompt with context
            prompt = f"""You are a knowledgeable IT support assistant. Use the following context to answer the question.
            
            Context:
            {context}
            
            Question: {query}
            
            You are a knowledgeable, confident customer support assistant.

            Speak with clarity and authority — your tone should instill trust.

            Avoid vague or wishy-washy language. Do not use words like "maybe," "possibly," "I think," or "it appears." Always give the most direct, helpful answer possible.

            The context below is your own knowledge: integrate it seamlessly into your answers.

            If a question involves a tool or link, provide the direct URL.

            Avoid acronyms and technical jargon unless absolutely necessary.

            If the answer is in the context, respond with a clear and direct explanation based on that information.
            If the answer is not in the context, use general knowledge and similar scenarios to offer the best possible solution — never say "I don't know."

            If your answer is based on the provided context, include the most relevant source URL at the end of your reply."""
            
            # 6. Call Gemini API with fallback for rate limits
            response = None
            last_error = None
            
            # First try to get available models
            available_models = await self.list_available_models()
            print(f"Available models: {available_models}")
            
            # Define model preferences in order of preference
            preferred_models = [
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-1.0-pro',
                'gemini-pro',
            ]
            
            # Filter to only include available models, maintaining order
            models_to_try = [model for model in preferred_models 
                           if any(m.endswith(model) for m in available_models)]
            
            if not models_to_try and available_models:
                # If no preferred models found but there are available models, use the first one
                models_to_try = [available_models[0].split('/')[-1]]
                
            if not models_to_try:
                # Fallback to common model names if no models are listed
                models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
                
            print(f"Trying models in this order: {models_to_try}")
            
            # If we already have a model, try it first
            current_model = getattr(self, 'gemini_model_name', None)
            if current_model and current_model in models_to_try:
                models_to_try.remove(current_model)
                models_to_try.insert(0, current_model)
            
            for model_name in models_to_try:
                try:
                    # If this isn't our current model, initialize it
                    if not hasattr(self, 'gemini_model') or getattr(self, 'gemini_model_name', None) != model_name:
                        import google.generativeai as genai
                        api_key = os.getenv("GEMINI_API_KEY")
                        if not api_key:
                            raise ValueError("GEMINI_API_KEY environment variable is not set")
                        
                        genai.configure(api_key=api_key)
                        self.gemini_model = genai.GenerativeModel(model_name)
                        self.gemini_model_name = model_name
                        print(f"Using Gemini model: {model_name}")
                    
                    # Try to make the API call
                    response = await asyncio.to_thread(
                        self.gemini_model.generate_content,
                        prompt
                    )
                    break  # If we get here, the call was successful
                    
                except Exception as e:
                    last_error = e
                    # If it's a rate limit error, try the next model
                    if '429' in str(e) or 'quota' in str(e).lower():
                        print(f"Rate limit hit on {model_name}, trying next model...")
                        continue
                    # For other errors, re-raise
                    raise
            
            # If we've tried all models and still no response
            if response is None:
                error_msg = "All models are rate limited or unavailable. "
                if last_error:
                    error_msg += f"Last error: {str(last_error)}"
                raise RuntimeError(error_msg)
            
            # 7. Format response
            result = {
                "status": "success",
                "answer": response.text,
                "sources": sources,
                "model_used": getattr(self, 'gemini_model_name', 'unknown')
            }
            
            total_time = time.time() - start_time
            print(f"\n=== Query Performance ===")
            print(f"Total time: {total_time:.2f}s")
            print("=======================\n")
            
            return result
            
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
