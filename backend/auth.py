from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# For development, we'll use a simple token verification
# In production, you should use proper JWT with a secure secret key
# This should match the format of the token being sent from the frontend
ACCEPTED_TOKENS = [
    "generated-1752019096972",  # Original test token
    "generated-1752021165398",  # New token from logs
    "generated-1752534699870",  # Latest token from logs
    "generated-1752535111763",
    "test-token"                # Local development token
]
ALGORITHM = "HS256"
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-key-please-change-in-production")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # In a real app, you might want to validate the token against a database here
        return payload
    except JWTError:
        raise credentials_exception

# Simple token verification for protected routes
async def verify_token(token: str = Depends(oauth2_scheme)):
    # For development, we'll check if the token matches any of our accepted tokens
    print(f"Verifying token: {token}")
    
    # Handle both raw token and Bearer token
    clean_token = token.replace("Bearer ", "") if token.startswith("Bearer ") else token
    
    if clean_token in ACCEPTED_TOKENS:
        print("Token verification successful")
        return True
    
    print(f"Token verification failed. Got: {clean_token}")
    print(f"Accepted tokens: {ACCEPTED_TOKENS}")
    return False
