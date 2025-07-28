from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class IndexStatus(str, Enum):
    CREATING = "creating"
    ACTIVE = "active"
    UPDATING = "updating"
    DELETING = "deleting"
    ERROR = "error"

class UserIndexBase(BaseModel):
    """Base model for index information"""
    name: str = Field(..., description="The full Pinecone index name")
    display_name: str = Field(..., description="User-friendly display name for the index")
    
    class Config:
        from_attributes = True

class UserIndexCreate(UserIndexBase):
    """Model for creating a new index"""
    pass

class UserIndexUpdate(BaseModel):
    """Model for updating an existing index"""
    display_name: Optional[str] = Field(
        None, 
        min_length=1, 
        max_length=100, 
        description="New display name for the index"
    )

class UserIndexInfo(UserIndexBase):
    """Complete model for index information including read-only fields"""
    document_count: int = Field(
        0, 
        description="Number of documents in the index"
    )
    created_at: datetime = Field(
        ..., 
        description="ISO 8601 timestamp when the index was created"
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="ISO 8601 timestamp when the index was last updated"
    )
    status: IndexStatus = Field(
        IndexStatus.ACTIVE,
        description="Current status of the index"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "castlerock-abc123-my-index",
                "display_name": "my-index",
                "document_count": 42,
                "created_at": "2023-01-01T12:00:00Z",
                "updated_at": "2023-01-02T15:30:00Z",
                "status": "active"
            }
        }

class IndexListResponse(BaseModel):
    """Response model for listing indices"""
    indices: List[UserIndexInfo]
    total: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "indices": [
                    {
                        "name": "castlerock-abc123-my-index",
                        "display_name": "my-index",
                        "document_count": 42,
                        "created_at": "2023-01-01T12:00:00Z",
                        "status": "active"
                    }
                ],
                "total": 1
            }
        }

class IndexOperationResponse(BaseModel):
    """Generic response model for index operations"""
    success: bool
    message: str
    index_name: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Index created successfully",
                "index_name": "castlerock-abc123-my-index"
            }
        }
