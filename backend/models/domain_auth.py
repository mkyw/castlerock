from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List
from datetime import datetime

class DomainLink(BaseModel):
    """Model for domain-API key mapping"""
    user_id: str = Field(..., description="The ID of the user who owns this domain")
    index_name: str = Field(..., description="The index this domain is linked to")
    domain: str = Field(..., description="The domain to be linked (e.g., example.com)")
    api_key: str = Field(..., description="API key for this domain")
    is_active: bool = Field(default=True, description="Whether this domain link is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this link was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When this link was last updated")
    description: Optional[str] = Field(None, description="Optional description for this domain link")

class CreateDomainLink(BaseModel):
    """Model for creating a new domain link"""
    index_name: str = Field(..., description="The index to link this domain to")
    domain: str = Field(..., description="The domain to be linked (e.g., example.com)")
    description: Optional[str] = Field(None, description="Optional description for this domain link")

class DomainLinkResponse(DomainLink):
    """Response model for domain links"""
    id: str = Field(..., description="Unique identifier for this domain link")
