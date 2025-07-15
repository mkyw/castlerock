import asyncio
import os
from dotenv import load_dotenv
from kb_rag_system import KBScraper

# Load environment variables
load_dotenv()

async def test_query(user_id, query="What is this knowledge base about?"):
    """Test querying with a specific user_id"""
    print(f"Testing query for user_id: {user_id}")
    
    try:
        # Create a KBScraper instance with the specified user_id
        scraper = KBScraper(user_id=user_id)
        
        # Query the knowledge base
        print(f"Sending query: '{query}'")
        result = await scraper.query(query)
        
        # Print the result
        print("\nQuery result:")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Answer: {result.get('answer', 'No answer')}")
        print(f"Sources: {result.get('sources', [])}")
        
        # Clean up
        await scraper.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # List of tokens to try
    tokens = [
        "generated-1752021165398",  # This one has data according to our check
        "generated-1752534699870",
        "generated-1752535111763",
    ]
    
    # Test each token
    for token in tokens:
        asyncio.run(test_query(token))
        print("\n" + "="*50 + "\n") 