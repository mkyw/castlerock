from fastapi import (
    FastAPI, UploadFile, File, HTTPException, Depends, status, 
    Request, Response, Header, Form, Query, Body
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr, HttpUrl, AnyHttpUrl
from typing import List, Dict, Any, Optional
import os
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta, timezone
import uvicorn
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Token expiration time in minutes

# Local imports
from auth import get_current_user, TokenData
from kb_rag_system import KBScraper
from utils.index_utils import (
    generate_index_name, 
    get_user_indices_util, 
    register_user_index, 
    update_index_metadata,
    get_index_metadata,
    delete_user_index,
    update_index_document_count
)
from pinecone_service import PineconeService
from utils.document_processor import DocumentProcessor, SUPPORTED_DOCUMENT_EXTENSIONS

# Database
from database import engine
from models.db_models import Base

# Authentication and security
from auth import (
    create_access_token, 
    create_refresh_token,
    verify_token,
    Token
)
from jose import JWTError, jwt
from fastapi.security import HTTPBearer
from auth import get_token_payload, oauth2_scheme

# Domain authentication
from middleware.domain_auth import DomainAuthMiddleware
from routers import domain_auth
from services.domain_auth_service import DomainAuthService

# Application components
# Remove duplicate imports as they're already imported above

# Models
from models.index_models import (
    UserIndexInfo,
    IndexListResponse,
    IndexOperationResponse,
    IndexStatus,
    UserIndexCreate,
    UserIndexUpdate
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Create database tables
Base.metadata.create_all(bind=engine)

# Include routers
app.include_router(domain_auth.router)

# Function to get all whitelisted domains for CORS
def get_all_whitelisted_domains():
    try:
        # Get all domain links
        domain_links = DomainAuthService._load_domain_links()
        
        # Extract unique domains
        domains = set()
        for link_id, link in domain_links.items():
            if link.get('is_active', True) and 'domain' in link:
                domain = link['domain'].split(':')[0].lower()
                # Add both http and https versions
                domains.add(f"http://{domain}")
                domains.add(f"https://{domain}")
                # Also add with www subdomain
                if not domain.startswith('www.'):
                    domains.add(f"http://www.{domain}")
                    domains.add(f"https://www.{domain}")
                # Add localhost for development
                domains.add("http://localhost:3000")
                domains.add("http://localhost:3001")
                domains.add("http://localhost:8000")
                
        # Always include default origins
        default_origins = [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:8000",
            "https://localhost",
            "https://localhost:3000"
        ]
        
        for origin in default_origins:
            domains.add(origin)
            
        return list(domains)
    except Exception as e:
        logger.error(f"Error getting whitelisted domains: {e}")
        # Return default origins if there's an error
        return [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:8000",
            "https://localhost",
            "https://localhost:3000"
        ]

# Add domain authentication middleware
app.add_middleware(
    DomainAuthMiddleware,
    auto_error=True  # Set to False if you want to allow unauthenticated requests
)

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
    allow_origins=get_all_whitelisted_domains(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
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
    """Request model for querying the knowledge base"""
    query: str = Field(..., description="The search query")
    k: int = Field(5, ge=1, le=20, description="Number of results to return")
    index_name: Optional[str] = Field(
        None, 
        description="Name of the index to query (defaults to user's default index)"
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None,
        description="List of previous messages in the conversation, each with 'role' and 'content'"
    )
    context_documents: Optional[List[str]] = Field(
        None,
        description="List of document contents to use as additional context"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is RAG?",
                "k": 5,
                "index_name": "castlerock-abc123-my-index",
                "conversation_history": [
                    {"role": "user", "content": "What are the different wifi networks?"},
                    {"role": "assistant", "content": "There are three networks: eduroam, UWNet, and WiscVPN."}
                ]
            }
        }

class ProcessWebsiteRequest(BaseModel):
    """Request model for processing a website"""
    url: HttpUrl = Field(..., description="URL of the website to process")
    index_name: Optional[str] = Field(
        None,
        description="Name of the index to store the processed content"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com",
                "index_name": "castlerock-abc123-my-index"
            }
        }

class ProcessPDFRequest(BaseModel):
    """Request model for processing a PDF file"""
    filename: str = Field(..., description="Name of the uploaded PDF file")
    index_name: Optional[str] = Field(
        None,
        description="Name of the index to store the processed content"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "filename": "research_paper.pdf",
                "index_name": "castlerock-abc123-my-index"
            }
        }

class ProcessDocumentRequest(BaseModel):
    """Request model for processing a document URL"""
    url: AnyHttpUrl = Field(..., description="URL of the document to process")
    index_name: Optional[str] = Field(
        None, 
        description="Name of the index to use (defaults to user's default index)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/document.pdf",
                "index_name": "my-knowledge-base"
            }
        }

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
# Authentication endpoints
class UserLogin(BaseModel):
    email: EmailStr
    password: str

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class UserResponse(BaseModel):
    id: str
    email: str
    is_active: bool = True
    created_at: datetime

@app.post("/auth/token", response_model=Token)
async def login_for_access_token(user_data: UserLogin):
    """
    Authenticate user and return access and refresh tokens.
    
    In a production environment, you would verify the username and password against a database.
    """
    # For demo purposes, we'll accept any valid email
    # In a real app, verify credentials against your user database
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_data.email, "email": user_data.email}, 
        expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token({"sub": user_data.email, "email": user_data.email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token
    }

@app.post("/auth/login", response_model=Dict[str, Any])
async def login(user_data: UserLogin):
    """
    Authenticate user and return access and refresh tokens with user info.
    
    This endpoint is used by the frontend for authentication.
    """
    # For demo purposes, we'll accept any valid email
    # In a real app, verify credentials against your user database
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_data.email, "email": user_data.email}, 
        expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token({"sub": user_data.email, "email": user_data.email})
    
    # Create a user object with the email as the ID
    user_id = user_data.email.replace("@", "_at_").replace(".", "_dot_")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token,
        "user": {
            "id": user_id,
            "email": user_data.email,
            "name": user_data.email.split('@')[0],
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    }

@app.post("/auth/refresh-token", response_model=Dict[str, str])
async def refresh_access_token(request: RefreshTokenRequest):
    """
    Refresh an access token using a refresh token.
    
    The refresh token should be sent in the request body.
    """
    try:
        # Verify the refresh token
        payload = jwt.decode(
            request.refresh_token, 
            os.getenv("JWT_SECRET_KEY"), 
            algorithms=["HS256"],
            options={"verify_aud": False}
        )
        
        # Check if token is a refresh token
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )
            
        # Check if token is expired
        if datetime.now(timezone.utc).timestamp() > payload.get("exp", 0):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired",
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": payload["sub"], "email": payload.get("email")}, 
            expires_delta=access_token_expires
        )
        
        # Create new refresh token
        new_refresh_token = create_refresh_token(data={"sub": payload["sub"]})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "refresh_token": new_refresh_token
        }
        
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token: Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )

@app.get("/core/users/me", response_model=UserResponse)
async def read_current_user(current_user: TokenData = Depends(get_current_user)):
    """
    Get the current authenticated user's information.
    
    In a real application, this would fetch the user's data from your database.
    """
    return {
        "id": current_user.sub,
        "email": current_user.email,
        "created_at": datetime.now(timezone.utc),
        "is_active": True
    }

@app.get("/core/users/", response_model=List[UserResponse])
async def list_users(current_user: TokenData = Depends(get_current_user)):
    """
    List all users (admin only).
    
    In a real application, this would require admin privileges and fetch users from your database.
    """
    # In a real app, you would check if the current user has admin privileges
    # and fetch users from your database
    # For now, we'll return a list with just the current user
    return [{
        "id": current_user.sub,
        "email": current_user.email,
        "created_at": datetime.now(timezone.utc),
        "is_active": True
    }]

# API Endpoints
@app.post("/api/rag/process/website")
async def process_website(
    request: ProcessWebsiteRequest,
    current_user: TokenData = Depends(get_current_user),
    authorization: str = Header(None)
):
    """Process a website and add it to the knowledge base"""
    if not current_user.sub:
        raise HTTPException(status_code=401, detail="Invalid user data in token")
    
    try:
        # Convert HttpUrl to string to avoid 'HttpUrl' object has no attribute 'decode' error
        url_str = str(request.url)
        logger.info(f"Processing website for user {current_user.sub}: {url_str}")
        
        # Get the user's scraper instance with the specified index name
        scraper = get_scraper(current_user.sub)
        
        # Update the index name if provided
        if request.index_name:
            scraper.update_index_name(request.index_name)
        
        # Call the RAG system's process_website method with the string URL
        result = await scraper.process_website(url_str)
        
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
    index_name: Optional[str] = Form(None),
    current_user: TokenData = Depends(get_current_user)
):
    """Process a PDF file and add it to the knowledge base"""
    if not current_user.sub:
        raise HTTPException(status_code=401, detail="Invalid user data in token")
    
    temp_file_path = None
    try:
        logger.info(f"Processing PDF for user {current_user.sub}: {file.filename}")
        
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        # Get the user's scraper instance with the specified index name
        scraper = get_scraper(current_user.sub)
        
        # Update the index name if provided
        if index_name:
            scraper.update_index_name(index_name)
        
        # Process the PDF
        result = await scraper.process_pdf(temp_file_path)
        
        return {
            "status": "success",
            "message": f"Successfully processed PDF: {file.filename}",
            "index_used": scraper.get_index_name()
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        # Clean up the temporary file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.error(f"Error removing temporary file {temp_file_path}: {str(e)}")

@app.post("/api/rag/process/document")
async def process_document_url(
    request: ProcessDocumentRequest,
    current_user: TokenData = Depends(get_current_user)
):
    """Process a document URL directly and add it to the knowledge base"""
    if not current_user.sub:
        raise HTTPException(status_code=401, detail="Invalid user data in token")
    
    try:
        # Convert AnyHttpUrl to string to avoid 'AnyHttpUrl' object has no attribute 'decode' error
        url_str = str(request.url)
        logger.info(f"Processing document URL for user {current_user.sub}: {url_str}")
        
        # Parse URL to get file extension
        parsed_url = urlparse(url_str)
        path = parsed_url.path.lower()
        
        # Check if this is a supported document type
        is_supported = any(path.endswith(ext) for ext in SUPPORTED_DOCUMENT_EXTENSIONS.keys())
        if not is_supported:
            supported_extensions = ", ".join(SUPPORTED_DOCUMENT_EXTENSIONS.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported document type. Supported types: {supported_extensions}"
            )
        
        # Get the user's scraper instance with the specified index name
        scraper = get_scraper(current_user.sub)
        
        # Update the index name if provided
        if request.index_name:
            scraper.update_index_name(request.index_name)
        
        # Start the document processor if not already running
        await scraper.document_processor.start_processing()
        
        # Queue the document for processing
        await scraper.document_processor.queue_document(url_str, parsed_url.netloc)
        
        return {
            "status": "success",
            "message": f"Document URL added to processing queue: {url_str}",
            "index_used": scraper.get_index_name()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error processing document URL: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/rag/indices", response_model=List[Dict[str, Any]])
@app.get("/api/rag/indices/", response_model=List[Dict[str, Any]])
async def list_indices(
    current_user: TokenData = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    List all indices for the current user with pagination support.
    
    Returns a paginated list of index metadata including the index name, display name,
    document count, and creation timestamp.
    """
    # Check if user is authenticated
    if current_user is None:
        logger.warning("Unauthorized access attempt to list_indices - no valid token provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    try:
        # Get the user's indices from our metadata store
        all_indices = get_user_indices_util(current_user.sub)
        
        # Apply pagination
        paginated_indices = all_indices[skip:skip + limit]
        
        # Convert to our response model
        indices = [
            {
                "name": idx.get('name', ''),
                "display_name": idx.get('display_name', 'Unnamed'),
                "document_count": idx.get('document_count', 0),
                "created_at": idx.get('created_at', datetime.now(timezone.utc).isoformat()),
                "updated_at": idx.get('updated_at'),
                "status": idx.get('status', 'active')
            }
            for idx in paginated_indices
        ]
        
        return indices
        
    except Exception as e:
        logger.error(f"Error listing indices: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user indices"
        )

@app.post("/api/rag/indices", response_model=Dict[str, Any])
@app.post("/api/rag/indices/", response_model=Dict[str, Any])
async def create_index_route(
    index_data: Dict[str, Any],
    current_user: TokenData = Depends(get_current_user)
):
    """
    Create a new index for the user.
    
    This will create a new Pinecone index with the specified display name.
    The actual index name will be generated to ensure uniqueness.
    """
    try:
        # Get the display name from the request or use a default
        display_name = index_data.get("display_name", "New Index").strip()
        
        # Generate a safe index name
        index_name, safe_display_name = generate_index_name(
            user_id=current_user.sub,
            display_name=display_name
        )
        
        logger.info(f"Creating index with name: {index_name}, display_name: {display_name}")
        
        # Register the index in our metadata store
        index_meta = register_user_index(
            user_id=current_user.sub,
            index_name=index_name,
            display_name=display_name  # Use the original display name, not the sanitized one
        )
        
        # Initialize the Pinecone service with the generated index name
        service = PineconeService(
            user_id=current_user.sub,
            index_name=index_name  # Use the generated index name directly
        )
        
        # Update status to creating
        update_index_metadata(
            index_name=index_name,
            updates={
                'status': 'creating',
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
        )
        
        try:
            # The index is already created during PineconeService initialization
            # No need to call service._create_index() again
            
            # Update status to active
            update_index_metadata(
                index_name=index_name,
                updates={
                    'status': 'active',
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Return the created index info
            return {
                "name": index_name,  # Return the actual index name used
                "display_name": display_name,  # Return the original display name
                "document_count": 0,
                "created_at": index_meta['created_at'],
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": "active"
            }
            
        except Exception as e:
            # Update status to error if creation fails
            update_index_metadata(
                index_name=index_name,
                updates={
                    'status': 'error',
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                    'error': str(e)
                }
            )
            raise
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating index: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create index: {str(e)}"
        )

@app.get(
    "/api/rag/indices/{index_name}",
    response_model=UserIndexInfo,
    summary="Get index details",
    description="Retrieve detailed information about a specific index"
)
@app.get(
    "/api/rag/indices/{index_name}/",
    response_model=UserIndexInfo,
    summary="Get index details",
    description="Retrieve detailed information about a specific index"
)
async def get_index(
    index_name: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Get detailed information about a specific index.
    
    This includes the index name, display name, document count, and status.
    """
    try:
        # Verify the index belongs to the user
        index_meta = get_index_metadata(current_user.sub, index_name)
        if not index_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Index not found or access denied"
            )
            
        return UserIndexInfo(
            name=index_meta['name'],
            display_name=index_meta.get('display_name', 'Unnamed'),
            document_count=index_meta.get('document_count', 0),
            created_at=index_meta.get('created_at', datetime.now(timezone.utc).isoformat()),
            updated_at=index_meta.get('updated_at'),
            status=index_meta.get('status', IndexStatus.ACTIVE)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting index details: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve index details"
        )

@app.patch(
    "/api/rag/indices/{index_name}",
    response_model=UserIndexInfo,
    summary="Update index",
    description="Update the display name or other metadata for an index"
)
async def update_index(
    index_name: str,
    updates: UserIndexUpdate,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Update an existing index's metadata.
    
    Currently only supports updating the display name.
    """
    try:
        # Verify the index belongs to the user
        index_meta = get_index_metadata(current_user.sub, index_name)
        if not index_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Index not found or access denied"
            )
            
        # Prepare updates
        update_data = {
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Only include fields that were provided in the request
        if updates.display_name is not None:
            update_data['display_name'] = updates.display_name
        
        # Update the index metadata
        updated_meta = update_index_metadata(index_name, update_data)
        
        return UserIndexInfo(
            name=updated_meta['name'],
            display_name=updated_meta.get('display_name', 'Unnamed'),
            document_count=updated_meta.get('document_count', 0),
            created_at=updated_meta.get('created_at', datetime.now(timezone.utc).isoformat()),
            updated_at=updated_meta.get('updated_at'),
            status=updated_meta.get('status', IndexStatus.ACTIVE)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating index: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update index: {str(e)}"
        )

@app.delete(
    "/api/rag/indices/{index_name}",
    response_model=IndexOperationResponse,
    summary="Delete index",
    description="Permanently delete an index and all its contents"
)
async def delete_index(
    index_name: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Delete a user's index.
    
    This will remove the index from Pinecone and clean up any associated metadata.
    Note: This operation is irreversible and will permanently delete all data in the index.
    """
    try:
        # First verify the index belongs to the user
        index_meta = get_index_metadata(current_user.sub, index_name)
        if not index_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Index not found or access denied"
            )
        
        # Update status to deleting
        update_index_metadata(
            index_name=index_name,
            updates={
                'status': IndexStatus.DELETING,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
        )
        
        try:
            # First remove from our metadata store to prevent any race conditions
            delete_user_index(current_user.sub, index_name)
            
            try:
                # Then delete from Pinecone
                service = PineconeService(
                    user_id=current_user.sub,
                    index_name=index_name
                )
                
                # Try to delete the index
                service.pc.delete_index(index_name)
                logger.info(f"Successfully deleted index {index_name} from Pinecone")
                
            except Exception as e:
                if "not found" not in str(e).lower():
                    logger.error(f"Error deleting index from Pinecone: {e}")
                    # We'll still consider this a success since we've removed our metadata
                    # This prevents getting stuck with orphaned metadata
            
            return IndexOperationResponse(
                success=True,
                message=f"Index '{index_meta.get('display_name', index_name)}' deleted successfully",
                index_name=index_name
            )
            
        except Exception as e:
            # Update status to error if deletion fails
            update_index_metadata(
                index_name=index_name,
                updates={
                    'status': IndexStatus.ERROR,
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                    'error': str(e)
                }
            )
            raise
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting index: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete index: {str(e)}"
        )

@app.post("/api/rag/query")
async def query_rag(
    request: Request,
    query_data: QueryRequest,
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme)
):
    """
    Query the knowledge base
    
    This endpoint supports both regular user authentication and domain-based authorization.
    Domain authorization is handled by the DomainAuthMiddleware, which checks if the request
    is coming from a whitelisted domain for the specified index.
    """
    logger.info(f"RAG query request: {query_data.dict()}")
    
    # Check for domain authentication first
    domain_user_id = getattr(request.state, 'domain_authenticated_user_id', None)
    domain_index_name = getattr(request.state, 'domain_index_name', None)
    
    # If domain authentication succeeded, use that user ID and index name
    if domain_user_id:
        logger.info(f"Using domain authenticated user ID: {domain_user_id}")
        user_id = domain_user_id
        
        # Use the domain index name if available, otherwise use the one from the request
        if domain_index_name:
            logger.info(f"Using index name from domain authentication: {domain_index_name}")
            index_name = domain_index_name
        else:
            index_name = query_data.index_name
    # Otherwise, check for JWT authentication
    elif authorization:
        try:
            # Manually validate the token
            token_data = await get_token_payload(authorization.credentials)
            if token_data and token_data.sub:
                logger.info(f"Using JWT authenticated user ID: {token_data.sub}")
                user_id = token_data.sub
                index_name = query_data.index_name
            else:
                logger.error("Invalid token data")
                raise HTTPException(status_code=401, detail="Invalid authentication token")
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid authentication token")
    else:
        # No authentication found
        logger.error("No valid authentication found")
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Initialize the Pinecone service with the specified index name or use default
        service = PineconeService(
            user_id=user_id,
            index_name=index_name
        )
        
        logger.info(f"Querying index: {service.index_name}")
        
        # Get the KBScraper instance for this user
        scraper = get_scraper(user_id)
        
        # Update the scraper's index name to match the query
        scraper.update_index_name(service.index_name)
        
        # Process the query using the scraper (which will use the LLM)
        logger.info(f"Processing query with LLM: {query_data.query}")
        
        # Check if we have conversation history
        if query_data.conversation_history:
            logger.info(f"Conversation history provided with {len(query_data.conversation_history)} messages")
        
        # Check if we have context documents
        if query_data.context_documents:
            logger.info(f"Context documents provided: {len(query_data.context_documents)}")
        
        llm_results = await scraper.query(
            query=query_data.query,
            k=query_data.k,
            conversation_history=query_data.conversation_history,
            context_documents=query_data.context_documents
        )
        
        logger.info(f"Query completed successfully for user {user_id}")
        logger.info(f"LLM model used: {llm_results.get('model_used', 'unknown')}")
        
        # Format the response
        return {
            "success": True,
            "index_used": service.index_name,
            "answer": llm_results.get("answer", "No answer found"),
            "sources": llm_results.get("sources", []),
            "model_used": llm_results.get("model_used", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        
        # Check if this is an empty index error
        if "'list' object has no attribute 'get'" in str(e):
            return {
                "success": False,
                "error_type": "empty_index",
                "answer": "This knowledge base is empty. Please add documents to the index before querying.",
                "sources": []
            }
            
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/debug/auth/token")
async def debug_token(current_user: TokenData = Depends(get_current_user)):
    """
    Debug endpoint to check JWT token validation
    """
    return {
        "sub": current_user.sub,
        "email": current_user.email,
        "exp": current_user.exp,
        "token_valid": True,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
