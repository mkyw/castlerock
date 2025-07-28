import os
import logging
import asyncio
import aiohttp
import tempfile
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

# Set up logging
logger = logging.getLogger(__name__)

# Document extensions we can process
SUPPORTED_DOCUMENT_EXTENSIONS = {
    # Documents
    '.pdf': 'application/pdf',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.txt': 'text/plain',
    '.rtf': 'application/rtf',
    
    # Presentations
    '.ppt': 'application/vnd.ms-powerpoint',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    
    # Spreadsheets
    '.xls': 'application/vnd.ms-excel',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    
    # Images - limited support
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    
    # Other
    '.csv': 'text/csv',
}

class DocumentProcessor:
    """Handles downloading and processing of various document types"""
    
    def __init__(self, user_id: str, scraper=None):
        """
        Initialize the document processor
        
        Args:
            user_id: User ID for the document processor
            scraper: Reference to the KBScraper instance for processing documents
        """
        self.user_id = user_id
        self.scraper = scraper
        self.download_queue = asyncio.Queue()
        self.processing_task = None
        self.shutdown_requested = False
        self.temp_dir = tempfile.mkdtemp(prefix="castlerock_docs_")
        logger.info(f"Created temporary directory for documents: {self.temp_dir}")
    
    def is_document_url(self, url: str) -> bool:
        """
        Check if a URL points to a supported document type
        
        Args:
            url: The URL to check
            
        Returns:
            bool: True if the URL points to a supported document
        """
        if not url:
            return False
            
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            # Check if the URL ends with any of our supported extensions
            return any(path.endswith(ext) for ext in SUPPORTED_DOCUMENT_EXTENSIONS.keys())
        except Exception as e:
            logger.error(f"Error checking document URL {url}: {e}")
            return False
    
    def get_document_type(self, url: str) -> Optional[str]:
        """
        Get the document type from a URL
        
        Args:
            url: The URL to check
            
        Returns:
            str: The document type extension (e.g., '.pdf')
        """
        if not url:
            return None
            
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            for ext in SUPPORTED_DOCUMENT_EXTENSIONS.keys():
                if path.endswith(ext):
                    return ext
            return None
        except Exception:
            return None
    
    async def start_processing(self):
        """Start the document processing task"""
        if self.processing_task is None:
            self.processing_task = asyncio.create_task(self._process_documents())
            logger.info("Document processor started")
    
    async def queue_document(self, url: str, base_domain: str) -> bool:
        """
        Add a document URL to the processing queue
        
        Args:
            url: The document URL to process
            base_domain: The base domain for the document (for reference)
            
        Returns:
            bool: True if the document was queued successfully
        """
        try:
            await self.download_queue.put((url, base_domain))
            logger.info(f"Queued document for processing: {url}")
            return True
        except Exception as e:
            logger.error(f"Error queuing document {url}: {e}")
            return False
    
    async def _process_documents(self):
        """Main document processing loop"""
        async with aiohttp.ClientSession() as session:
            while not self.shutdown_requested:
                try:
                    # Get the next document from the queue with a timeout
                    try:
                        url, base_domain = await asyncio.wait_for(self.download_queue.get(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # No documents in queue, continue waiting
                        continue
                    
                    logger.info(f"Processing document: {url}")
                    
                    try:
                        # Download and process the document
                        await self._download_and_process(session, url, base_domain)
                    except Exception as e:
                        logger.error(f"Error processing document {url}: {e}")
                    finally:
                        # Mark the task as done
                        self.download_queue.task_done()
                        
                except asyncio.CancelledError:
                    logger.info("Document processor task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in document processor: {e}")
                    await asyncio.sleep(1)  # Avoid tight loop on errors
        
        logger.info("Document processor stopped")
    
    async def _download_and_process(self, session: aiohttp.ClientSession, url: str, base_domain: str):
        """
        Download and process a document
        
        Args:
            session: The aiohttp session to use for downloading
            url: The document URL to download
            base_domain: The base domain for the document
        """
        doc_type = self.get_document_type(url)
        if not doc_type:
            logger.warning(f"Unknown document type for URL: {url}")
            return
        
        # Create a temporary file with the correct extension
        filename = os.path.basename(urlparse(url).path)
        if not filename:
            filename = f"document_{hash(url)}{doc_type}"
        
        temp_path = os.path.join(self.temp_dir, filename)
        
        try:
            # Download the document
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status != 200:
                    logger.error(f"Failed to download document {url}: HTTP {response.status}")
                    return
                
                # Save the document to a temporary file
                with open(temp_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                
                logger.info(f"Downloaded document to {temp_path}")
                
                # Process the document based on its type
                if doc_type in ['.pdf', '.txt']:
                    # We can process these directly
                    await self._process_text_document(temp_path, url)
                elif doc_type in ['.doc', '.docx', '.rtf', '.ppt', '.pptx', '.xls', '.xlsx']:
                    # Office documents require conversion
                    await self._process_office_document(temp_path, url, doc_type)
                elif doc_type in ['.jpg', '.jpeg', '.png']:
                    # Images require OCR
                    await self._process_image_document(temp_path, url)
                else:
                    logger.warning(f"No processor available for document type {doc_type}")
        
        except Exception as e:
            logger.error(f"Error processing document {url}: {e}")
        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.error(f"Error removing temporary file {temp_path}: {e}")
    
    async def _process_text_document(self, file_path: str, url: str):
        """Process a text document (PDF, TXT)"""
        if self.scraper is None:
            logger.error("No scraper available for processing document")
            return
        
        try:
            if file_path.lower().endswith('.pdf'):
                # Use the existing PDF processor
                result = await self.scraper.process_pdf(file_path)
                logger.info(f"Processed PDF document {url}: {result}")
            elif file_path.lower().endswith('.txt'):
                # Process text file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                # Add the document to the knowledge base
                await self.scraper.process_document(url, text)
                logger.info(f"Processed text document {url}")
        except Exception as e:
            logger.error(f"Error processing text document {url}: {e}")
    
    async def _process_office_document(self, file_path: str, url: str, doc_type: str):
        """Process an Office document (DOC, DOCX, PPT, PPTX, XLS, XLSX)"""
        # This requires additional libraries like python-docx, python-pptx, etc.
        # For now, we'll just log that we need to implement this
        logger.info(f"Office document processing not yet implemented for {doc_type}: {url}")
        
        # TODO: Implement document processing based on type
        # For example:
        # if doc_type in ['.doc', '.docx']:
        #     # Use python-docx
        #     import docx
        #     doc = docx.Document(file_path)
        #     text = '\n'.join([p.text for p in doc.paragraphs])
        #     await self.scraper.process_document(url, text)
    
    async def _process_image_document(self, file_path: str, url: str):
        """Process an image document with OCR (JPG, PNG)"""
        # This requires additional libraries like pytesseract
        # For now, we'll just log that we need to implement this
        logger.info(f"Image document processing not yet implemented: {url}")
        
        # TODO: Implement OCR for images
        # For example:
        # try:
        #     import pytesseract
        #     from PIL import Image
        #     text = pytesseract.image_to_string(Image.open(file_path))
        #     await self.scraper.process_document(url, text)
        # except ImportError:
        #     logger.error("pytesseract not installed. Install with: pip install pytesseract")
    
    async def shutdown(self):
        """Shut down the document processor"""
        self.shutdown_requested = True
        
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Clean up temporary directory
        try:
            for filename in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, filename))
            os.rmdir(self.temp_dir)
            logger.info(f"Removed temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {e}") 