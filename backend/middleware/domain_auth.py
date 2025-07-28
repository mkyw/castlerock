from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import re
from typing import Optional, Dict, Any, List, Tuple
import json
from urllib.parse import urlparse, parse_qs
import logging

from services.domain_auth_service import DomainAuthService

# Configure logging
logger = logging.getLogger(__name__)

class DomainAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware for domain-based authorization.
    
    This middleware verifies that requests to specific indices are coming from
    whitelisted domains. It checks the referer/origin headers against the
    registered domains for each index.
    """
    
    def __init__(self, app, auto_error: bool = True):
        self.auto_error = auto_error
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        # Skip validation for certain paths
        if self.should_skip_validation(request.url.path):
            return await call_next(request)
        
        # Get domain from referer or origin header
        domain = self.get_domain_from_request(request)
        
        # If we have a domain, try to find the associated index
        if domain:
            try:
                # Get all domain links
                domain_links = self._load_domain_links_for_domain(domain)
                
                if domain_links:
                    # Use the first matching domain link
                    domain_link = domain_links[0]
                    index_name = domain_link.get('index_name')
                    user_id = domain_link.get('user_id')
                    
                    if index_name and user_id:
                        # Store the index_name and user_id in the request state
                        request.state.domain_authenticated_user_id = user_id
                        request.state.domain_index_name = index_name
                        logger.info(f"Domain {domain} authenticated for index {index_name}, user {user_id}")
                
            except Exception as e:
                logger.error(f"Error in domain authentication: {str(e)}")
        
        # Continue to the next middleware/route
        response = await call_next(request)
        return response
        
    def _load_domain_links_for_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Load domain links that match the given domain"""
        all_domain_links = DomainAuthService._load_domain_links()
        matching_links = []
        
        for link_id, link in all_domain_links.items():
            if not link.get('is_active', True):
                continue
                
            if self.is_matching_domain(domain, link.get('domain', '')):
                matching_links.append(link)
                
        return matching_links

    async def is_domain_authorized(self, domain: str, index_name: str) -> bool:
        """Check if the domain is authorized for this index"""
        # Get all domain links for this index
        domain_links = DomainAuthService.get_domain_links_by_index_name(index_name)
        
        # If no domain links exist, allow access (no restrictions set)
        if not domain_links:
            return True
            
        # Check if any domain link matches this domain
        for link in domain_links:
            if link.get('is_active', True) and self.is_matching_domain(domain, link.get('domain', '')):
                return True
                
        return False

    def should_skip_validation(self, path):
        # Skip validation for OPTIONS requests (preflight) and public endpoints
        if path in ['/auth/login', '/auth/refresh-token', '/debug/auth/token']:
            return True
        
        # Skip domain validation for non-API routes
        if not path.startswith('/api/'):
            return True
            
        # Skip domain validation for indices management routes (these use JWT auth)
        if path.startswith('/api/rag/indices'):
            return True
            
        return False

    def get_domain_from_request(self, request):
        # Get domain from referer header
        referer = request.headers.get('referer')
        if referer:
            return urlparse(referer).netloc
        
        # Get domain from origin header
        origin = request.headers.get('origin')
        if origin:
            return urlparse(origin).netloc
        
        return None

    async def get_index_name_from_request(self, request):
        """Extract index name from request path or body"""
        path = request.url.path
        
        # Check if index name is in path (for specific index operations)
        path_match = re.search(r'/api/rag/indices/([^/]+)', path)
        if path_match:
            return path_match.group(1)
        
        # For query endpoints, check the request body
        if path == '/api/rag/query' and request.method == 'POST':
            try:
                content_type = request.headers.get("content-type", "")
                if "application/json" in content_type:
                    body = await request.body()
                    if body:
                        body_data = json.loads(body)
                        index_name = body_data.get("index_name")
                        # Store the body back in the request for the endpoint to use
                        request._body = body
                        return index_name
            except json.JSONDecodeError:
                pass
        
        return None

    def is_matching_domain(self, domain, expected_domain):
        """Check if domain matches or is a subdomain of expected_domain"""
        if not domain or not expected_domain:
            return False
            
        # Remove port number from domains
        domain = domain.split(':')[0].lower()
        expected_domain = expected_domain.split(':')[0].lower()
        
        # Check for exact match
        if domain == expected_domain:
            return True
            
        # Check if domain is a subdomain of expected_domain
        if domain.endswith('.' + expected_domain):
            return True
            
        # Check if expected_domain is a subdomain of domain
        if expected_domain.endswith('.' + domain):
            return True
            
        return False
