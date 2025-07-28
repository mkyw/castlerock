from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# JWT Configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours (60 min * 24)
REFRESH_TOKEN_EXPIRE_DAYS = 7
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable is not set")

# Token models
class TokenData(BaseModel):
    sub: str  # user ID
    email: Optional[str] = None
    exp: Optional[int] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: str

oauth2_scheme = HTTPBearer(auto_error=False)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a new JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create a new JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_token_payload(token: str) -> TokenData:
    """Verify and decode a JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            logger.warning("Token missing 'sub' claim")
            raise credentials_exception
        return TokenData(**payload)
    except JWTError as e:
        logger.warning(f"JWT validation error: {str(e)}")
        raise credentials_exception

async def get_current_user(token: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme)) -> Optional[TokenData]:
    """Dependency to get the current user from the JWT token"""
    if not token:
        logger.debug("No token provided in request")
        return None
    try:
        return await get_token_payload(token.credentials)
    except HTTPException as e:
        logger.warning(f"Token validation failed: {e.detail}")
        return None

async def verify_token(token: HTTPAuthorizationCredentials = Depends(oauth2_scheme)) -> bool:
    """Verify that the token is valid"""
    try:
        await get_token_payload(token.credentials)
        return True
    except HTTPException:
        return False
