import json
import os
import secrets
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

from fastapi import HTTPException, status

# Configure logging
logger = logging.getLogger(__name__)

# Path to the JSON file for storing domain links
DOMAIN_LINKS_FILE = Path("data/domain_links.json")

# Create data directory if it doesn't exist
os.makedirs(DOMAIN_LINKS_FILE.parent, exist_ok=True)

class DomainAuthService:
    """Service for managing domain-based authentication"""
    
    @staticmethod
    def _load_domain_links() -> Dict[str, Dict]:
        """Load domain links from JSON file"""
        if not DOMAIN_LINKS_FILE.exists():
            return {}
            
        try:
            with open(DOMAIN_LINKS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    @staticmethod
    def _save_domain_links(domain_links: Dict[str, Dict]) -> None:
        """Save domain links to JSON file"""
        with open(DOMAIN_LINKS_FILE, 'w') as f:
            json.dump(domain_links, f, indent=2, default=str)
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure API key"""
        return f"sk_domain_{secrets.token_urlsafe(32)}"
    
    @classmethod
    def normalize_domain(cls, domain: str) -> str:
        """Normalize domain by removing protocol, path, and port"""
        if not domain:
            return ''
            
        # Remove protocol if present
        domain = domain.lower()
        if '://' in domain:
            domain = domain.split('://', 1)[1]
            
        # Remove path and query parameters
        domain = domain.split('/')[0].split('?')[0].split('#')[0]
        
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':', 1)[0]
            
        # Remove leading/trailing dots
        domain = domain.strip('.')
        
        # Basic validation
        if not domain or len(domain) > 253 or '..' in domain:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid domain format"
            )
            
        return domain
        
    @classmethod
    def is_subdomain(cls, domain: str, parent_domain: str) -> bool:
        """Check if domain is a subdomain of parent_domain"""
        if not domain or not parent_domain:
            return False
            
        # Handle exact match
        if domain == parent_domain:
            return True
            
        # Handle subdomains (e.g., sub.example.com is a subdomain of example.com)
        return domain.endswith(f".{parent_domain}")
    
    @classmethod
    def create_domain_link(
        cls, 
        user_id: str, 
        index_name: str, 
        domain: str, 
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new domain link for an index"""
        try:
            domain = cls.normalize_domain(domain)
            domain_links = cls._load_domain_links()
            
            # Check if domain is already linked to any index
            for link_id, link in domain_links.items():
                if not link.get('is_active', True):
                    continue
                    
                # Check for domain conflict
                domain_conflict = (
                    domain == link.get('domain') or 
                    cls.is_subdomain(domain, link.get('domain', '')) or 
                    cls.is_subdomain(link.get('domain', ''), domain)
                )
                
                # Allow same domain for different indices, but not same domain for same index
                if domain_conflict and link.get('index_name') == index_name:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Domain {domain} is already linked to this index"
                    )
            
            # Create new domain link
            link_id = f"dom_{secrets.token_urlsafe(12)}"
            api_key = cls.generate_api_key()
            created_at = datetime.utcnow()
            
            domain_link = {
                'id': link_id,
                'user_id': user_id,
                'index_name': index_name,
                'domain': domain,
                'api_key': api_key,
                'is_active': True,
                'created_at': created_at.isoformat(),
                'updated_at': created_at.isoformat(),
                'description': description
            }
            
            domain_links[link_id] = domain_link
            cls._save_domain_links(domain_links)
            
            return domain_link
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create domain link: {str(e)}"
            )
        
    @classmethod
    def get_domain_link(cls, link_id: str) -> Optional[Dict[str, Any]]:
        """Get a domain link by ID"""
        domain_links = cls._load_domain_links()
        link = domain_links.get(link_id)
        if link and link.get('is_active', True):
            return link
        return None
        
    @classmethod
    def get_domain_link_by_api_key(cls, api_key: str) -> Optional[Dict[str, Any]]:
        """Get a domain link by API key"""
        domain_links = cls._load_domain_links()
        for link in domain_links.values():
            if link.get('api_key') == api_key and link.get('is_active', True):
                return link
        return None
        
    @classmethod
    def get_user_domain_links(cls, user_id: str) -> List[Dict[str, Any]]:
        """Get all domain links for a user"""
        domain_links = cls._load_domain_links()
        return [
            link for link in domain_links.values() 
            if link.get('user_id') == user_id and link.get('is_active', True)
        ]
        
    @classmethod
    def get_domain_links_by_index(
        cls, 
        user_id: str, 
        index_name: str
    ) -> List[Dict[str, Any]]:
        """Get all domain links for a user's index"""
        domain_links = cls._load_domain_links()
        return [
            link for link in domain_links.values()
            if link.get('user_id') == user_id and 
               link.get('index_name') == index_name and 
               link.get('is_active', True)
        ]
    
    @classmethod
    def get_domain_links_by_index_name(cls, index_name: str) -> List[Dict[str, Any]]:
        """Get all domain links for an index regardless of user"""
        domain_links = cls._load_domain_links()
        return [
            link for link in domain_links.values()
            if link.get('index_name') == index_name and 
               link.get('is_active', True)
        ]
        
    @classmethod
    def delete_domain_link(cls, user_id: str, link_id: str) -> bool:
        """Delete a domain link (soft delete)"""
        try:
            domain_links = cls._load_domain_links()
            
            # Verify the index exists and belongs to the user
            link = domain_links.get(link_id)
            if not link or link.get('user_id') != user_id or not link.get('is_active', True):
                return False
            
            # Mark as inactive instead of deleting
            link['is_active'] = False
            link['updated_at'] = datetime.utcnow().isoformat()
            domain_links[link_id] = link
            cls._save_domain_links(domain_links)
            
            logger.info(f"Deleted domain link: {link_id} for index {link.get('index_name')}")
            return True
        except Exception as e:
            logger.error(f"Error deleting domain link: {e}", exc_info=True)
            return False
