from fastapi import APIRouter, Depends, HTTPException, status, Query, Request, Header
from typing import List, Optional, Annotated
import logging

from models.domain_auth import CreateDomainLink, DomainLinkResponse
from services.domain_auth_service import DomainAuthService
from auth import get_token_payload, HTTPBearer

# Set up logging
logger = logging.getLogger(__name__)

# Security scheme for API key
oauth2_scheme = HTTPBearer()

# Dependency to get current user from JWT token
async def get_current_user(authorization: str = Header(...)):
    """Dependency to get the current user from the Authorization header"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Use 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = authorization.split(" ")[1]
    try:
        payload = await get_token_payload(token)
        return payload
    except HTTPException as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during authentication: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during authentication"
        )

router = APIRouter(prefix="/api/domain-auth", tags=["domain-auth"])

@router.post("/domains", response_model=DomainLinkResponse)
async def create_domain_link(
    domain_data: CreateDomainLink,
    authorization: Annotated[str, Header()] = None
):
    """
    Create a new domain link for a specific index.
    
    This will generate an API key that can be used to authenticate requests from the specified domain.
    """
    try:
        # Get current user from Authorization header
        current_user = await get_current_user(authorization)
        
        logger.info(f"Creating domain link for user {current_user.sub} - {domain_data.domain}")
        
        domain_link = DomainAuthService.create_domain_link(
            user_id=current_user.sub,
            index_name=domain_data.index_name,
            domain=domain_data.domain,
            description=domain_data.description
        )
        
        logger.info(f"Successfully created domain link for {domain_data.domain}")
        return domain_link
        
    except HTTPException as e:
        logger.error(f"HTTP error creating domain link: {str(e.detail)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating domain link: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create domain link: {str(e)}"
        )

# Dependency to validate the index name against the domain link
def validate_index_name(
    request: Request,
    index_name: str = Query(..., description="The name of the index")
) -> str:
    """Validate that the index name matches the domain link's index"""
    domain_link = getattr(request.state, 'domain_link', None)
    if not domain_link:
        # If no domain link is found, this is an internal API call, so we can skip validation
        return index_name
        
    if index_name != domain_link['index_name']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API key not valid for index: {index_name}"
        )
    return index_name

@router.get("/domains", response_model=List[DomainLinkResponse])
async def list_domain_links(
    index_name: Optional[str] = Query(None, description="Filter by index name"),
    authorization: Annotated[str, Header()] = None
):
    """List domain links for the authenticated user, optionally filtered by index"""
    try:
        # Get current user from Authorization header
        current_user = await get_current_user(authorization)
        
        logger.info(f"Listing domain links for user {current_user.sub}")
        
        if index_name:
            links = DomainAuthService.get_domain_links_by_index(
                current_user.sub, 
                index_name
            )
        else:
            links = DomainAuthService.get_user_domain_links(current_user.sub)
            
        logger.info(f"Found {len(links)} domain links for user {current_user.sub}")
        return links
        
    except HTTPException as e:
        logger.error(f"HTTP error listing domain links: {str(e.detail)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error listing domain links: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve domain links: {str(e)}"
        )

@router.delete("/domains/{link_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_domain_link(
    link_id: str,
    authorization: Annotated[str, Header()] = None
):
    """Delete a domain link (soft delete by marking as inactive)"""
    try:
        # Get current user from Authorization header
        current_user = await get_current_user(authorization)
        
        logger.info(f"Deleting domain link {link_id} for user {current_user.sub}")
        
        success = DomainAuthService.delete_domain_link(
            current_user.sub, 
            link_id
        )
        
        if not success:
            logger.warning(f"Domain link {link_id} not found or permission denied for user {current_user.sub}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Domain link not found or you don't have permission to delete it"
            )
            
        logger.info(f"Successfully deleted domain link {link_id}")
        return None
        
    except HTTPException as e:
        logger.error(f"HTTP error deleting domain link: {str(e.detail)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting domain link: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete domain link: {str(e)}"
        )
