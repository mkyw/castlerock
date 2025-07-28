import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
INDEX_NAME_PREFIX = "castlerock"
INDEX_METADATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "data"
)
INDEX_METADATA_FILE = os.path.join(INDEX_METADATA_DIR, "user_indices.json")

# Ensure the metadata directory exists
os.makedirs(INDEX_METADATA_DIR, exist_ok=True)

def get_user_hash(user_id: str, length: int = 8) -> str:
    """Generate a consistent hash for a user ID.
    
    Args:
        user_id: The user's unique identifier (email)
        length: Length of the hash to return (max 64 for SHA-256)
        
    Returns:
        A hexadecimal hash of the user ID
    """
    # Ensure length is within bounds
    length = max(1, min(64, length))
    # Create a SHA-256 hash of the user ID
    hash_obj = hashlib.sha256(user_id.encode('utf-8'))
    # Return the requested number of characters
    return hash_obj.hexdigest()[:length]

def _ensure_metadata_file() -> None:
    """Create the metadata file if it doesn't exist"""
    if not os.path.exists(INDEX_METADATA_FILE):
        with open(INDEX_METADATA_FILE, 'w') as f:
            json.dump({"indices": {}}, f, indent=2)

def _load_metadata() -> Dict[str, Any]:
    """Load the metadata from the JSON file"""
    _ensure_metadata_file()
    try:
        with open(INDEX_METADATA_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading metadata file: {e}")
        # Return a default structure if the file is corrupted
        return {"indices": {}}

def _save_metadata(metadata: Dict[str, Any]) -> None:
    """Save the metadata to the JSON file"""
    try:
        with open(INDEX_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error saving metadata file: {e}")
        raise

def generate_index_name(user_id: str, display_name: str) -> Tuple[str, str]:
    """
    Generate a safe index name from a user ID and display name.
    
    Args:
        user_id: The user's unique ID
        display_name: The user-provided display name for the index
        
    Returns:
        A tuple of (full_index_name, safe_display_name)
    """
    try:
        # Create a hash of the user ID for the prefix
        user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:8]
        
        # Create a URL-safe version of the display name
        safe_name = re.sub(r'[^a-zA-Z0-9-]', '-', display_name.lower().strip())
        safe_name = re.sub(r'-+', '-', safe_name).strip('-')
        
        # Ensure the name isn't empty
        if not safe_name:
            safe_name = "unnamed"
        
        # Truncate if too long (Pinecone has a 45 char limit for index names)
        max_length = 45 - len(f"{INDEX_NAME_PREFIX}-{user_hash}-")
        if len(safe_name) > max_length:
            safe_name = safe_name[:max_length]
        
        # Ensure the name is unique by appending a counter if needed
        base_name = safe_name
        counter = 1
        
        while True:
            full_name = f"{INDEX_NAME_PREFIX}-{user_hash}-{safe_name}"
            
            # Check if this index name already exists
            metadata = _load_metadata()
            if full_name not in metadata.get("indices", {}):
                break
                
            # If it exists, try the next number
            safe_name = f"{base_name}-{counter}"
            counter += 1
        
        return full_name, safe_name
    except Exception as e:
        logger.error(f"Error generating index name: {e}")
        raise ValueError(f"Failed to generate index name: {str(e)}")

def register_user_index(user_id: str, index_name: str, display_name: str) -> Dict[str, Any]:
    """
    Register a new user index in the metadata store.
    
    Args:
        user_id: The user's unique ID
        index_name: The full Pinecone index name
        display_name: The user-friendly display name
        
    Returns:
        The created index metadata
    """
    try:
        metadata = _load_metadata()
        now = datetime.utcnow().isoformat()
        
        index_data = {
            "name": index_name,
            "display_name": display_name,
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
            "document_count": 0,
            "status": "creating"
        }
        
        if "indices" not in metadata:
            metadata["indices"] = {}
        
        metadata["indices"][index_name] = index_data
        _save_metadata(metadata)
        
        logger.info(f"Registered new index: {index_name} for user {user_id}")
        return index_data
    except Exception as e:
        logger.error(f"Error registering index: {e}", exc_info=True)
        raise ValueError(f"Failed to register index: {str(e)}")

def get_user_indices(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all indices for a specific user.
    
    Args:
        user_id: The user's unique ID
        
    Returns:
        A list of index metadata dictionaries, sorted by most recently created first
    """
    try:
        metadata = _load_metadata()
        user_indices = [
            idx for idx in metadata.get("indices", {}).values()
            if idx.get("user_id") == user_id
        ]
        
        # Sort by creation date (newest first)
        return sorted(
            user_indices,
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )
    except Exception as e:
        logger.error(f"Error getting user indices: {e}", exc_info=True)
        return []

def get_index_metadata(user_id: str, index_name: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific index if it belongs to the user.
    
    Args:
        user_id: The user's unique ID
        index_name: The full Pinecone index name
        
    Returns:
        The index metadata if found and accessible, None otherwise
    """
    try:
        metadata = _load_metadata()
        index_data = metadata.get("indices", {}).get(index_name)
        
        if index_data and index_data.get("user_id") == user_id:
            return index_data
        return None
    except Exception as e:
        logger.error(f"Error getting index metadata: {e}", exc_info=True)
        return None

def update_index_metadata(
    index_name: str, 
    updates: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Update metadata for an index.
    
    Args:
        index_name: The full Pinecone index name
        updates: Dictionary of fields to update
        
    Returns:
        The updated index metadata if found, None otherwise
    """
    try:
        metadata = _load_metadata()
        
        if index_name not in metadata.get("indices", {}):
            return None
        
        # Update the specified fields
        for key, value in updates.items():
            if key in ["name", "user_id"]:
                continue  # Don't allow changing these fields
            metadata["indices"][index_name][key] = value
        
        # Always update the updated_at timestamp
        metadata["indices"][index_name]["updated_at"] = datetime.utcnow().isoformat()
        
        _save_metadata(metadata)
        
        logger.debug(f"Updated metadata for index {index_name}")
        return metadata["indices"][index_name]
    except Exception as e:
        logger.error(f"Error updating index metadata: {e}", exc_info=True)
        raise ValueError(f"Failed to update index metadata: {str(e)}")

def delete_user_index(user_id: str, index_name: str) -> bool:
    """
    Delete an index from the metadata store.
    
    Note: This does not delete the actual Pinecone index, just the metadata.
    
    Args:
        user_id: The user's unique ID
        index_name: The full Pinecone index name
        
    Returns:
        True if the index was deleted, False if not found or not owned by user
    """
    try:
        metadata = _load_metadata()
        
        # Verify the index exists and belongs to the user
        index_data = metadata.get("indices", {}).get(index_name)
        if not index_data or index_data.get("user_id") != user_id:
            return False
        
        # Remove the index from metadata
        del metadata["indices"][index_name]
        _save_metadata(metadata)
        
        logger.info(f"Deleted index metadata: {index_name} for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting user index: {e}", exc_info=True)
        return False

def update_index_document_count(
    index_name: str, 
    change: int = 1
) -> bool:
    """
    Update the document count for an index.
    
    Args:
        index_name: The full Pinecone index name
        change: The change in document count (can be negative)
        
    Returns:
        True if the update was successful, False otherwise
    """
    try:
        metadata = _load_metadata()
        
        if index_name not in metadata.get("indices", {}):
            logger.warning(f"Index {index_name} not found when updating document count")
            return False
        
        # Update the document count
        current_count = metadata["indices"][index_name].get("document_count", 0)
        new_count = max(0, current_count + change)
        metadata["indices"][index_name]["document_count"] = new_count
        metadata["indices"][index_name]["updated_at"] = datetime.utcnow().isoformat()
        
        _save_metadata(metadata)
        
        logger.debug(f"Updated document count for {index_name}: {current_count} -> {new_count}")
        return True
    except Exception as e:
        logger.error(f"Error updating document count: {e}", exc_info=True)
        return False

# For backward compatibility with existing imports
get_user_indices_util = get_user_indices
delete_user_index_util = delete_user_index
