from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os
import logging
from pathlib import Path
import asyncio
from dotenv import load_dotenv
from auth import verify_token, oauth2_scheme

# Load environment variables from .env file
load_dotenv()

# Import the KB scraper implementation
from kb_rag_system import KBScraper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Dictionary to store user-specific scrapers
user_scrapers = {}

def get_scraper(user_id: str) -> KBScraper:
    """Get or create a KBScraper instance for the user"""
    # Ensure we're using the actual token as the user_id for Pinecone
    if user_id not in user_scrapers:
        print(f"Creating new scraper for user: {user_id}")
        user_scrapers[user_id] = KBScraper(
            user_id=user_id,
            max_pages=1000000
        )
    return user_scrapers[user_id]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Knowledge Base RAG API",
        "endpoints": [
            {
                "path": "/api/rag/query",
                "method": "POST",
                "description": "Query the knowledge base",
                "required_headers": ["Authorization: Bearer <token>"]
            },
            {
                "path": "/api/rag/process/website",
                "method": "POST",
                "description": "Process a website and add it to the knowledge base",
                "required_headers": ["Authorization: Bearer <token>"]
            },
            {
                "path": "/api/rag/process/pdf",
                "method": "POST",
                "description": "Process a PDF file and add it to the knowledge base",
                "required_headers": ["Content-Type: multipart/form-data", "Authorization: Bearer <token>"]
            }
        ]
    }

# Request models
class QueryRequest(BaseModel):
    query: str
    k: int = 5

class ProcessWebsiteRequest(BaseModel):
    url: str

class ProcessPDFRequest(BaseModel):
    filename: str

# Startup event
@app.on_event("startup")
async def startup_event():
    print("Starting up KB RAG API...")
    print("API is ready to handle requests")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down KB RAG API...")
    # Close all user scrapers with timeout
    for user_id, scraper in list(user_scrapers.items()):
        try:
            if hasattr(scraper, 'close'):
                try:
                    # Use timeout to avoid hanging
                    await asyncio.wait_for(scraper.close(), timeout=10.0)
                    print(f"Closed scraper for user {user_id}")
                except asyncio.TimeoutError:
                    print(f"Timeout closing scraper for user {user_id}, forcing shutdown")
                    # Force cleanup
                    if hasattr(scraper, '_cleanup_embeddings'):
                        scraper._cleanup_embeddings()
                    scraper._is_closed = True
                    scraper.shutdown_requested = True
        except Exception as e:
            print(f"Error closing scraper for user {user_id}: {e}")
    
    # Force cleanup of any remaining resources
    user_scrapers.clear()
    print("Shutdown complete")

# API Endpoints
@app.post("/api/rag/process/website")
async def process_website(
    request: ProcessWebsiteRequest,
    authorization: str = Header(None)
):
    """Process a website and add it to the knowledge base"""
    # Verify token and get user ID
    token = authorization.split(" ")[-1] if authorization else ""
    if not await verify_token(token):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    
    # Use the token directly as the user_id for consistent index naming
    user_id = token
    
    try:
        logger.info(f"Processing website for user {user_id}: {request.url}")
        
        # Get the user's scraper instance
        scraper = get_scraper(user_id)
        
        # Call the RAG system's process_website method
        result = await scraper.process_website(request.url)
        
        # Ensure the response has the expected format
        if not isinstance(result, dict) or 'status' not in result:
            result = {
                "status": "success" if not result.get('error') else "error",
                "message": result.get('message', 'Website processing completed'),
                **result
            }
            
        logger.info(f"Website processing result: {result}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error processing website: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/rag/process/pdf")
async def process_pdf(
    file: UploadFile = File(...),
    authorization: str = Header(None)
):
    """Process a PDF file and add it to the knowledge base"""
    # Verify token and get user ID
    token = authorization.split(" ")[-1] if authorization else ""
    if not await verify_token(token):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    
    # Use the token directly as the user_id for consistent index naming
    user_id = token
    
    try:
        logger.info(f"Processing PDF for user {user_id}: {file.filename}")
        
        # Save the uploaded file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
            try:
                # Get the user's scraper instance
                scraper = get_scraper(user_id)
                
                # Process the PDF
                result = await scraper.process_pdf(file_path)
                return result
                
            finally:
                # Clean up the temporary file in all cases
                if os.path.exists(file_path):
                    os.remove(file_path)
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()

@app.post("/api/rag/query")
async def query_rag(
    request: QueryRequest,
    authorization: str = Header(None)
):
    """Query the knowledge base"""
    # Verify token and get user ID
    token = authorization.split(" ")[-1] if authorization else ""
    if not await verify_token(token):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    
    # Use the token directly as the user_id for consistent index naming
    user_id = token
    
    try:
        logger.info(f"Processing query for user {user_id}: {request.query}")
        
        # Get the user's scraper instance
        scraper = get_scraper(user_id)
        
        # Call the RAG system's query method
        results = await scraper.query(request.query, request.k)
        
        # Log the raw results for debugging
        logger.debug(f"Raw query results: {results}")
        
        # If the RAG system returned an error, handle it
        if isinstance(results, dict) and results.get('status') == 'error':
            return results
        
        # If we got a direct answer from the RAG system, return it as is
        if isinstance(results, dict) and 'answer' in results:
            return {
                'status': 'success',
                'answer': results['answer'],
                'sources': results.get('sources', [])
            }
            
        # Handle the case where we have a list of results
        results_list = []
        if isinstance(results, dict) and 'results' in results:
            results_list = results['results']
        elif isinstance(results, list):
            results_list = results
        
        # Format each result to match the expected frontend format
        formatted_results = []
        for item in results_list:
            if isinstance(item, dict):
                formatted_results.append({
                    'content': item.get('content', str(item)),
                    'source': item.get('source', 'system'),
                    'type': item.get('type', 'text'),
                    'score': item.get('score', 0.0)
                })
            else:
                formatted_results.append({
                    'content': str(item),
                    'source': 'system',
                    'type': 'text',
                    'score': 0.0
                })
        
        # Format the results as an answer
        response = {
            'status': 'success',
            'answer': '\n'.join([r['content'] for r in formatted_results]) if formatted_results else 'No results found.',
            'sources': list(set(r['source'] for r in formatted_results if r.get('source') and r['source'] != 'system'))
        }
        
        logger.info(f"Query processed successfully. Found {len(formatted_results)} results.")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"Error processing query: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {
            'status': 'error',
            'message': f"Failed to process query: {str(e)}",
            'results': []
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
