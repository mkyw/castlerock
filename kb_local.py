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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

# Configuration
CONFIG = {
    "max_concurrent_requests": 10,  # Increased concurrent requests
    "max_embedding_workers": 5,      # Increased embedding workers
    "request_timeout": 15,           # Slightly longer timeout for slower pages
    "batch_size": 20,                # Larger batch size for efficiency
    "save_interval": 10,             # Save less frequently for better performance
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "max_depth": 10,                 # Maximum link depth to follow
    "min_content_length": 500,        # Minimum content length to be considered valid
    "max_pages": 2000,               # Maximum number of pages to crawl
    "domain_whitelist": ["kb.wisc.edu"],  # Only crawl this domain
    "path_blacklist": [
        "/search.php", 
        "/search/",
        "/tag/",
        "/category/",
        "/author/",
        "/feed/",
        "/wp-",
        "/xmlrpc.php",
        "/trackback/"
    ],
    "deprecated_indicators": [
        "deprecated",
        "outdated",
        "no longer maintained",
        "archived",
        "legacy",
        "this page has been moved",
        "this content is no longer available"
    ]
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
    def __init__(self, base_url: str = "https://kb.wisc.edu/", max_pages: int = None, persist_dir: str = "faiss_index"):
        """
        Initialize the KB Scraper with FAISS vector store
        
        Args:
            base_url: The base URL to start scraping from
            max_pages: Maximum number of pages to crawl (None for no limit)
            persist_dir: Directory to save the FAISS index
        """
        self.base_url = base_url
        self.max_pages = max_pages or CONFIG["max_pages"]
        self.persist_dir = Path(persist_dir)
        self.visited_urls: Set[str] = set()
        self.pages_to_visit = deque([base_url])
        self.documents_processed = 0
        self.batches_processed = 0
        
        # Initialize embeddings
        print("Loading local embedding model (this may take a minute on first run)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Lightweight but effective model
            model_kwargs={"device": "cpu"},  # Use 'cuda' if you have a GPU
            encode_kwargs={"normalize_embeddings": False}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Initialize FAISS vector store
        self.vectorstore: Optional[FAISS] = None
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._load_vectorstore()
        
        # Initialize async queues
        self.doc_queue: asyncio.Queue[Optional[Document]] = asyncio.Queue()
        self.done_event = asyncio.Event()
        self.tasks: Set[asyncio.Task] = set()
        self.shutdown_requested = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _load_vectorstore(self) -> None:
        """Load or initialize the FAISS vector store"""
        index_path = self.persist_dir / "index.faiss"
        
        if index_path.exists():
            try:
                print("Loading existing FAISS index...")
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
        
        # Create a new empty FAISS index
        print("Creating a new FAISS index...")
        self.vectorstore = FAISS.from_texts(
            texts=["Initial document"],
            embedding=self.embeddings,
            metadatas=[{"source": "system", "timestamp": datetime.now().isoformat()}]
        )
        print("Created a new FAISS index with one document.")
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        if self.shutdown_requested:
            print("\nForce shutdown requested. Exiting immediately.")
            import sys
            sys.exit(1)
            
        print("\nShutdown requested. Finishing current tasks...")
        self.shutdown_requested = True
        self.done_event.set()
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and should be crawled"""
        # Skip if we've already visited this URL
        if url in self.visited_urls:
            return False
            
        # Parse the URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
                
            # Only crawl the target domain
            if not any(domain in parsed.netloc for domain in CONFIG["domain_whitelist"]):
                return False
                
            # Skip blacklisted paths
            path = parsed.path.lower()
            if any(blacklisted in path for blacklisted in CONFIG["path_blacklist"]):
                return False
                
            # Allow more URL patterns
            if not (path.endswith('/') or 
                   path.endswith('.html') or 
                   path.endswith('.php') or
                   '?' in path):  # Allow query params
                return False
                
            return True
            
        except ValueError:
            return False
    
    async def _fetch_page(self, session: aiohttp.ClientSession, url: str) -> CrawlResult:
        """Fetch a single page asynchronously"""
        try:
            headers = {
                'User-Agent': CONFIG["user_agent"],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': self.base_url,
            }
            
            print(f"  Fetching: {url}")
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=CONFIG["request_timeout"])) as response:
                if response.status != 200:
                    print(f"  Error: HTTP {response.status}")
                    return CrawlResult(url=url, content=None, links=[], error=f"HTTP {response.status}")
                    
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' not in content_type:
                    print(f"  Error: Unsupported content type: {content_type}")
                    return CrawlResult(url=url, content=None, links=[], error=f"Unsupported content type: {content_type}")
                
                html = await response.text()
                
                # Check if we got a valid HTML response
                if not html or '<html' not in html.lower():
                    print("  Error: Invalid HTML response")
                    return CrawlResult(url=url, content=None, links=[], error="Invalid HTML response")
                
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract main content - KB specific selectors
                main_content = (soup.select_one('div.page-content') or 
                              soup.select_one('article') or 
                              soup.select_one('div#content') or
                              soup.select_one('div.main-content') or
                              soup.find('main') or 
                              soup.find('div', class_='content') or
                              soup.body or soup)  # Fall back to body or entire doc
                
                # Remove unwanted elements
                for element in main_content.select('script, style, nav, footer, header, aside, .nav, .sidebar, .breadcrumb, .pagination'):
                    element.decompose()
                    
                # Get text with better spacing
                content = '\n'.join(line.strip() for line in main_content.get_text('\n').splitlines() if line.strip())
                
                # Skip pages with too little content
                if len(content) < CONFIG["min_content_length"]:
                    print(f"  Skipping: Content too short ({len(content)} chars)")
                    return CrawlResult(url=url, content=None, links=[], error="Content too short")
                
                # Check if page is deprecated
                if self._is_deprecated(content):
                    print("  Skipping: Page marked as deprecated")
                    return CrawlResult(url=url, content=None, links=[], error="Page marked as deprecated")
                
                # Extract links
                links = self._process_links(url, html)
                print(f"  Found {len(links)} links on page")
                
                return CrawlResult(url=url, content=content, links=links)
                
        except asyncio.TimeoutError:
            print("  Error: Request timeout")
            return CrawlResult(url=url, content=None, links=[], error="Request timeout")
        except Exception as e:
            print(f"  Error: {str(e)}")
            return CrawlResult(url=url, content=None, links=[], error=str(e))
    
    def _is_deprecated(self, content: str) -> bool:
        """Check if page content indicates it's deprecated"""
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in CONFIG["deprecated_indicators"])
        
    def _is_directory_page(self, soup: BeautifulSoup) -> bool:
        """
        Check if page is a directory/listing page that shouldn't be saved.
        Returns True only for pages that are purely navigation/directory listings.
        """
        # First, check for pages that definitely aren't directories
        # Pages with these elements are considered content pages, not directories
        content_indicators = [
            'article',
            'main-content',
            'content-body',
            'entry-content',
            'post-content',
            'page-content',
            'main-article',
            'article-body',
            'content-main',
            'main-content-area',
            'content-area',
            'main-article-content',
            'article-content',
            'node-content',
            'field-body',
            'field-content',
            'content-inner',
            'content-wrapper',
            'content-container',
            'content-main-wrapper',
            'content-main-inner',
            'content-main-container'
        ]
        
        # If we find any content indicators, it's not a directory page
        for element in soup.find_all(True, class_=True):
            classes = ' '.join(element.get('class', [])).lower()
            if any(f' {indicator} ' in f' {classes} ' for indicator in content_indicators):
                return False
                
        # Now check for definitive directory indicators
        directory_indicators = [
            'parent-directory',
            'directory-listing',
            'file-listing',
            'file-list',
            'file-index',
            'file-browser',
            'file-manager',
            'file-archive',
            'file-directory',
            'file-navigation',
            'dir-listing',
            'dir-list',
            'dir-index',
            'dir-browser',
            'dir-manager',
            'dir-archive',
            'dir-navigation',
            'folder-listing',
            'folder-list',
            'folder-index',
            'folder-browser',
            'folder-manager',
            'folder-archive',
            'folder-directory',
            'folder-navigation'
        ]
        
        # Check for directory indicators in class names
        for element in soup.find_all(True, class_=True):
            classes = ' '.join(element.get('class', [])).lower()
            if any(f' {indicator} ' in f' {classes} ' for indicator in directory_indicators):
                return True
                
        # Check for directory indicators in IDs
        for element in soup.find_all(id=True):
            if any(f' {indicator} ' in f' {element["id"].lower()} ' for indicator in directory_indicators):
                return True
                
        # Check for directory indicators in text
        text = soup.get_text().lower()
        directory_phrases = [
            'index of',
            'directory listing',
            'parent directory',
            'file list',
            'file index',
            'file browser',
            'file manager',
            'file archive',
            'file directory',
            'file navigation',
            'directory contents',
            'list of files',
            'file listing',
            'directory of files',
            'file directory listing',
            'directory listing of',
            'list of directories',
            'directory structure',
            'folder contents',
            'list of folders'
        ]
        
        # Only consider it a directory page if the phrase appears in the first few lines
        first_few_lines = '\n'.join(text.split('\n')[:10])
        if any(phrase in first_few_lines for phrase in directory_phrases):
            return True
            
        # Check link-to-text ratio (few text, many links)
        links = soup.find_all('a')
        text_length = len(text.strip())
        
        # Only consider it a directory page if there's very little text and many links
        if links and text_length < 300 and len(links) > 15:
            # But make sure it's not just a table of contents at the top of a content page
            main_content = soup.find('main') or soup.find('article') or soup.find('div', role='main')
            if main_content and len(main_content.get_text().strip()) > 1000:
                return False
            return True
            
        return False
        text_length = len(soup.get_text())
        
    def _process_links(self, base_url: str, html: str) -> List[str]:
        """Extract and normalize links from HTML"""
        if not html:
            return []
            
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = set()
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href'].strip()
                if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                    continue
                    
                # Handle relative URLs
                full_url = urljoin(base_url, href)
                
                # Normalize URL
                parsed = urlparse(full_url)
                
                # Only keep HTTP/HTTPS URLs
                if parsed.scheme not in ('http', 'https'):
                    continue
                
                # Remove fragments and query params for normalization
                full_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                full_url = full_url.rstrip('/')  # Normalize trailing slashes
                
                # Only include whitelisted domains
                if not any(domain in parsed.netloc for domain in CONFIG["domain_whitelist"]):
                    continue
                    
                # Skip blacklisted paths
                if any(blacklisted in parsed.path.lower() for blacklisted in CONFIG["path_blacklist"]):
                    continue
                    
                # Only include valid URLs
                if not self._is_valid_url(full_url):
                    continue
                    
                # Skip common file extensions
                path_lower = parsed.path.lower()
                if any(path_lower.endswith(ext) for ext in [
                    '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.doc', '.docx',
                    '.xls', '.xlsx', '.ppt', '.pptx', '.mp3', '.mp4', '.avi', '.mov',
                    '.css', '.js', '.ico', '.svg', '.woff', '.woff2', '.ttf', '.eot'
                ]):
                    continue
                    
                # Add to our set of links
                links.add(full_url)
            
            return list(links)
            
        except Exception as e:
            print(f"Error processing links: {e}")
            return []
    
    async def _process_url(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, url: str, depth: int):
        """Process a single URL with the semaphore"""
        async with sem:
            try:
                # Fetch the page
                print(f"\n[{depth}] Fetching: {url}")
                
                # First, check if the URL is valid before fetching
                if not self._is_valid_url(url):
                    print(f"Skipping invalid URL: {url}")
                    return
                
                    
                # Process the page content
                doc = Document(
                    page_content=result.content,
                    metadata={
                        "source": url,
                        "title": url.split('/')[-1] or "Untitled",
                        "last_modified": datetime.now().isoformat()
                    }
                )
                
                # Add to document queue for processing
                if not self.doc_queue.full():
                    await self.doc_queue.put(doc)
                
                # Process links if we haven't reached max depth
                if depth < CONFIG["max_depth"] and hasattr(result, 'links') and result.links:
                    for link in result.links:
                        if (link not in self.visited_urls and 
                            link not in self.pages_to_visit and
                            len(self.visited_urls) + len(self.pages_to_visit) < self.max_pages):
                            self.pages_to_visit.append(link)
                
                return result
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    async def crawl_worker(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore):
        """Worker that processes URLs from the queue"""
        while not self.shutdown_requested and (self.pages_to_visit or len(self.visited_urls) == 0):
            try:
                if not self.pages_to_visit:
                    if len(self.visited_urls) > 0:  # Only sleep if we've processed at least one page
                        print(f"{asyncio.current_task().get_name()}: Queue is empty, waiting...")
                        await asyncio.sleep(1)
                    continue
                    
                url = self.pages_to_visit.popleft()
                
                if url in self.visited_urls:
                    continue
                    
                print(f"\nProcessing: {url}")
                
                # Process the URL
                result = await self._process_url(session, sem, url, 0)
                
                if result and hasattr(result, 'links') and result.links:
                    for link in result.links:
                        if (link not in self.visited_urls and 
                            link not in self.pages_to_visit and 
                            len(self.visited_urls) + len(self.pages_to_visit) < self.max_pages):
                            self.pages_to_visit.append(link)
                
                # Mark as visited
                self.visited_urls.add(url)
                
                # Check if we've reached the maximum number of pages
                if len(self.visited_urls) >= self.max_pages:
                    print(f"Reached maximum number of pages ({self.max_pages}). Stopping...")
                    self.shutdown_requested = True
                    break
                    
            except asyncio.CancelledError:
                print("Crawler task was cancelled")
                break
            except Exception as e:
                print(f"Error in crawl_worker: {e}")
                import traceback
                traceback.print_exc()
                continue
        
def _process_links(self, base_url: str, html: str) -> List[str]:
    """Extract and normalize links from HTML"""
    if not html:
        return []
            
    try:
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
            
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
                    
            # Handle relative URLs
            full_url = urljoin(base_url, href)
                
            # Normalize URL
            parsed = urlparse(full_url)
                
            # Only keep HTTP/HTTPS URLs
            if parsed.scheme not in ('http', 'https'):
                continue
                
            # Remove fragments and query params for normalization
            full_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            full_url = full_url.rstrip('/')  # Normalize trailing slashes
                
            # Only include whitelisted domains
            if not any(domain in parsed.netloc for domain in CONFIG["domain_whitelist"]):
                continue
                    
            # Skip blacklisted paths
            if any(blacklisted in parsed.path.lower() for blacklisted in CONFIG["path_blacklist"]):
                continue
                    
            # Only include valid URLs
            if not self._is_valid_url(full_url):
                continue
                    
            # Skip common file extensions
            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in [
                '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.doc', '.docx',
                '.xls', '.xlsx', '.ppt', '.pptx', '.mp3', '.mp4', '.avi', '.mov',
                '.css', '.js', '.ico', '.svg', '.woff', '.woff2', '.ttf', '.eot'
            ]):
                continue
                    
            # Add to our set of links
            links.add(full_url)
            
        return list(links)
            
    except Exception as e:
        print(f"Error processing links: {e}")
        return []
    
async def _process_url(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, url: str, depth: int):
    """Process a single URL with the semaphore"""
    async with sem:
        try:
            # Fetch the page
            print(f"\n[{depth}] Fetching: {url}")
                
            # First, check if the URL is valid before fetching
            if not self._is_valid_url(url):
                print(f"Skipping invalid URL: {url}")
                return
                
                    
            # Process the page content
            doc = Document(
                page_content=result.content,
                metadata={
                    "source": url,
                    "title": url.split('/')[-1] or "Untitled",
                    "last_modified": datetime.now().isoformat()
                }
            )
                
            # Add to document queue for processing
            if not self.doc_queue.full():
                await self.doc_queue.put(doc)
                
            # Process links if we haven't reached max depth
            if depth < CONFIG["max_depth"] and hasattr(result, 'links') and result.links:
                for link in result.links:
                    if (link not in self.visited_urls and 
                        link not in self.pages_to_visit and
                        len(self.visited_urls) + len(self.pages_to_visit) < self.max_pages):
                        self.pages_to_visit.append(link)
                
            return result
                
        except Exception as e:
            print(f"Error processing {url}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
async def crawl_worker(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore):
    """Worker that processes URLs from the queue"""
    while not self.shutdown_requested and (self.pages_to_visit or len(self.visited_urls) == 0):
        try:
            if not self.pages_to_visit:
                if len(self.visited_urls) > 0:  # Only sleep if we've processed at least one page
                    print(f"{asyncio.current_task().get_name()}: Queue is empty, waiting...")
                    await asyncio.sleep(1)
                continue
                    
            url = self.pages_to_visit.popleft()
                
            if url in self.visited_urls:
                continue
                    
            print(f"\nProcessing: {url}")
                
            # Process the URL
            result = await self._process_url(session, sem, url, 0)
                
            if result and hasattr(result, 'links') and result.links:
                for link in result.links:
                    if (link not in self.visited_urls and 
                        link not in self.pages_to_visit and 
                        len(self.visited_urls) + len(self.pages_to_visit) < self.max_pages):
                        self.pages_to_visit.append(link)
                
            # Mark as visited
            self.visited_urls.add(url)
                
            # Check if we've reached the maximum number of pages
            if len(self.visited_urls) >= self.max_pages:
                print(f"Reached maximum number of pages ({self.max_pages}). Stopping...")
                self.shutdown_requested = True
                break
                    
        except asyncio.CancelledError:
            print("Crawler task was cancelled")
            break
        except Exception as e:
            print(f"Error in crawl_worker: {e}")
            import traceback
            traceback.print_exc()
            continue
    
async def crawl(self):
    """
    Main crawling method with improved crawling logic
    """
    print("\n" + "="*50)
    print(f"Starting crawl of {self.base_url}")
    print(f"Max pages: {self.max_pages}")
    print(f"Max depth: {CONFIG['max_depth']}")
    print(f"Max concurrent requests: {CONFIG['max_concurrent_requests']}")
    print("="*50 + "\n")
        
    # Start time for performance tracking
    start_time = time.time()
        
    # Initialize HTTP session with connection pooling
    connector = aiohttp.TCPConnector(
        force_close=True,
        enable_cleanup_closed=True,
        limit=CONFIG["max_concurrent_requests"],
        ttl_dns_cache=300  # 5 minute DNS cache TTL
    )
        
    timeout = aiohttp.ClientTimeout(total=CONFIG["request_timeout"])
        
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={
            'User-Agent': CONFIG["user_agent"],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
    ) as session:
        # Create semaphore to limit concurrent requests
        sem = asyncio.Semaphore(CONFIG["max_concurrent_requests"])
            
        # Create worker tasks
        workers = [
            asyncio.create_task(self.crawl_worker(session, sem), name=f"Worker-{i}")
            for i in range(min(10, CONFIG["max_concurrent_requests"]))  # Max 10 workers
        ]
            
        # Start embedding workers
        embed_workers = [
            asyncio.create_task(self.embedding_worker(i), name=f"Embedder-{i}")
            for i in range(min(3, CONFIG["max_embedding_workers"]))  # Max 3 embedders
        ]
            
        # Create a task to monitor progress
        progress_task = asyncio.create_task(self._progress_monitor())
            
        # Wait for all workers to complete
        try:
            # Wait for either all workers to complete or a keyboard interrupt
            done, pending = await asyncio.wait(
                workers + embed_workers,
                return_when=asyncio.FIRST_COMPLETED
            )
                
            # If we get here, either we're done or there was an error
            for task in done:
                if task.exception():
                    print(f"\nWorker {task.get_name()} failed: {task.exception()}")
                    import traceback
                    traceback.print_exception(type(task.exception()), task.exception(), task.get_stack())
                
        except asyncio.CancelledError:
            print("\nCrawl cancelled by user")
        except Exception as e:
            print(f"\nUnexpected error during crawl: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Signal workers to shut down
            self.shutdown_requested = True
            progress_task.cancel()
                
            # Cancel all workers
            for task in workers + embed_workers + [progress_task]:
                if not task.done():
                    task.cancel()
                
            # Wait for all tasks to complete
            await asyncio.gather(*workers + embed_workers + [progress_task], return_exceptions=True)
                
            # Close the document queue
            for _ in range(len(embed_workers)):
                await self.doc_queue.put(None)
                
            # Save the final vector store
            if self.vectorstore is not None:
                print("\nSaving final vector store...")
                await self.save_vectorstore_async()
        
    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print("Crawl completed!" if not self.shutdown_requested else "Crawl stopped by user")
    print(f"Pages crawled: {len(self.visited_urls)}")
    print(f"Document chunks processed: {self.documents_processed}")
    print(f"Batches processed: {self.batches_processed}")
    print(f"Total vectors in store: {self.get_vector_count()}")
    print(f"Time taken: {elapsed:.2f} seconds")
    print("="*50 + "\n")
    
async def _progress_monitor(self):
    """Periodically print crawl progress"""
    start_time = time.time()
    last_count = 0
        
    while not self.shutdown_requested:
        try:
            await asyncio.sleep(10)  # Print every 10 seconds
                
            current_time = time.time()
            elapsed = current_time - start_time
                
            # Calculate pages processed in the last interval
            processed = len(self.visited_urls)
            processed_diff = processed - last_count
            last_count = processed
                
            # Calculate rate
            rate = processed / elapsed if elapsed > 0 else 0
                
            print(f"\n=== Progress ===")
            print(f"Pages crawled: {processed}")
            print(f"In queue: {len(self.pages_to_visit)}")
            print(f"Rate: {rate:.2f} pages/sec")
            print(f"Elapsed: {elapsed:.1f} seconds")
            if self.pages_to_visit:
                print(f"Next: {self.pages_to_visit[0]}")
            print("================\n")
                
            # Auto-save every minute if we've processed documents
            if self.documents_processed > 0 and int(elapsed) % 60 == 0:
                print("Auto-saving vector store...")
                await self.save_vectorstore_async()
                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in progress monitor: {e}")
            continue
                
async def embedding_worker(self, worker_id: int) -> None:
    """Worker that processes documents and adds them to the vector store"""
    print(f"Embedding worker {worker_id} started")
    
    batch = []
    last_save = time.time()
    
    while not self.shutdown_requested:
        try:
            # Get a document with timeout to allow for periodic saves
            try:
                doc = await asyncio.wait_for(self.doc_queue.get(), timeout=1.0)
                if doc is None:  # None is a signal to shut down
                    break
                
                batch.append(doc)
                self.doc_queue.task_done()
                
                # Process batch if we have enough documents or it's time to save
                current_time = time.time()
                if (len(batch) >= CONFIG["batch_size"] or 
                    current_time - last_save >= CONFIG["save_interval"] * 60):
                    
                    if batch:  # Only process if we have documents
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
                                  f"(total vectors: {self.vectorstore.index.ntotal if self.vectorstore else 0})")
                            
                            # Save periodically
                            if current_time - last_save >= CONFIG["save_interval"] * 60:
                                await self.save_vectorstore_async()
                                last_save = current_time
                            
                            batch = []
                            
                        except Exception as e:
                            print(f"Error in embedding worker {worker_id}: {e}")
            
            except asyncio.TimeoutError:
                # Just continue the loop on timeout
                continue
                
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
    
    async def save_vectorstore_async(self) -> None:
        """Save the FAISS index asynchronously"""
        if not self.vectorstore:
            return
            
        try:
            # Save to a temporary file first, then rename to be atomic
            temp_path = self.persist_dir / "index.tmp"
            self.vectorstore.save_local(folder_path=str(temp_path), index_name="index")
            
            # If we get here, save was successful, so we can remove any existing index
            index_files = list(self.persist_dir.glob("index.*"))
            for f in index_files:
                if f.name != "index.tmp":
                    try:
                        if f.is_dir():
                            import shutil
                            shutil.rmtree(f)
                        else:
                            f.unlink()
                    except Exception as e:
                        print(f"Warning: Could not remove old index file {f}: {e}")
            
            # Move temp files to their final location
            for f in temp_path.glob("*"):
                try:
                    f.rename(self.persist_dir / f.name)
                except Exception as e:
                    print(f"Error moving {f} to destination: {e}")
            
            # Remove the temp directory
            try:
                temp_path.rmdir()
            except Exception as e:
                print(f"Warning: Could not remove temp directory {temp_path}: {e}")
                
            print(f"Saved vector store with {self.vectorstore.index.ntotal} vectors")
            
        except Exception as e:
            print(f"Error saving vector store: {e}")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search the vector store for similar documents
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_vector_count(self) -> int:
        """Get the total number of vectors in the store"""
        if not self.vectorstore or not hasattr(self.vectorstore, 'index'):
            return 0
        return self.vectorstore.index.ntotal


async def main():
    # Initialize the scraper
    scraper = KBScraper(
        base_url="https://kb.wisc.edu/",
        max_pages=1000000,  # Set to 0 for unlimited
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
