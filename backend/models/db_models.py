from sqlalchemy import Column, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base

class DomainLinkDB(Base):
    """Database model for domain links"""
    __tablename__ = "domain_links"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    index_name = Column(String, index=True, nullable=False)
    domain = Column(String, index=True, nullable=False)
    api_key = Column(String, unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    description = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "index_name": self.index_name,
            "domain": self.domain,
            "api_key": self.api_key,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "description": self.description
        }
