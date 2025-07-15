import os
from dotenv import load_dotenv
from pinecone import Pinecone
import sys

# Load environment variables
load_dotenv()

def check_pinecone():
    # Get config from environment
    api_key = os.getenv("PINECONE_API_KEY")
    
    if not api_key:
        print("Error: PINECONE_API_KEY environment variable is not set")
        sys.exit(1)
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)
    
    # List all indexes
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes]
    print(f"Available indexes: {index_names}")
    
    # Check if specific index was provided
    specific_index = None
    if len(sys.argv) > 1:
        specific_index = sys.argv[1]
        if specific_index not in index_names:
            print(f"Specified index '{specific_index}' does not exist!")
            specific_index = None
    
    # Check all indexes or just the specified one
    indexes_to_check = [specific_index] if specific_index else index_names
    
    for index_name in indexes_to_check:
        print(f"\n{'='*50}")
        print(f"Checking index: {index_name}")
        print(f"{'='*50}")
        
        # Connect to the index
        index = pc.Index(index_name)
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
        
        if stats.get('total_vector_count', 0) == 0:
            print("\nThe index is empty!")
            continue
        
        print(f"\nThe index contains {stats.get('total_vector_count')} vectors.")
        
        # Query a sample vector to check content
        try:
            # Create a simple random vector for testing
            import numpy as np
            test_vector = np.random.rand(384).tolist()  # Assuming 384 dimensions
            
            # Query the index
            results = index.query(vector=test_vector, top_k=5, include_metadata=True)
            
            print(f"\nSample query returned {len(results.get('matches', []))} results")
            
            # Print first result if available
            if results.get('matches') and len(results['matches']) > 0:
                first_match = results['matches'][0]
                print("\nFirst match metadata:")
                print(f"ID: {first_match.id}")
                print(f"Score: {first_match.score}")
                if hasattr(first_match, 'metadata') and first_match.metadata:
                    print(f"Text: {first_match.metadata.get('text', 'No text')[:100]}...")
                    print(f"Source: {first_match.metadata.get('source', 'No source')}")
                else:
                    print("No metadata available")
            else:
                print("No matches found in the query results")
        except Exception as e:
            print(f"Error querying index: {e}")

if __name__ == "__main__":
    check_pinecone() 