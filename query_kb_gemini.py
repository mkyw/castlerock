import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Load environment variables
load_dotenv()

class KBQueryGemini:
    def __init__(self, persist_dir: str = "faiss_index"):
        """Initialize the KB query system with Gemini"""
        self.persist_dir = Path(persist_dir)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectorstore = self._load_vectorstore()
        
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Try to use gemini-1.5-flash, which is the recommended model
        try:
            model_name = 'gemini-1.5-flash'
            print(f"Using model: {model_name}")
            self.model = genai.GenerativeModel(model_name)
            # Test the model with a simple prompt to verify it works
            self.model.generate_content("Test")
        except Exception as e:
            print(f"Error with gemini-1.5-flash: {str(e)}")
            print("Falling back to listing available models...")
            
            # Fallback: List available models if the default fails
            try:
                available_models = [m for m in genai.list_models() 
                                 if 'generateContent' in m.supported_generation_methods
                                 and 'vision' not in m.name.lower()]  # Skip vision models
                
                if not available_models:
                    raise Exception("No available text generation models found")
                
                # Use the first available text model
                model_name = available_models[0].name
                print(f"Falling back to model: {model_name}")
                self.model = genai.GenerativeModel(model_name)
            except Exception as e:
                raise Exception(f"Failed to initialize any Gemini model: {str(e)}")
        
    def _load_vectorstore(self) -> FAISS:
        """Load the FAISS vector store"""
        if not (self.persist_dir / "index.faiss").exists():
            raise FileNotFoundError(f"No FAISS index found at {self.persist_dir}")
        
        print("Loading FAISS vector store...")
        return FAISS.load_local(
            folder_path=str(self.persist_dir),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
            index_name="index"
        )

    def _format_context(self, docs) -> str:
        """Format the context from retrieved documents"""
        context = ""
        for i, doc in enumerate(docs, 1):
            context += f"Document {i} (Source: {doc.metadata['source']}):\n"
            context += doc.page_content + "\n\n"
        return context.strip()

    def query(self, question: str) -> str:
        """Query the knowledge base with a question using Gemini"""
        print(f"\nSearching for: {question}")
        
        try:
            # Retrieve relevant documents
            docs = self.vectorstore.similarity_search(question, k=5)
            
            if not docs:
                return "No relevant documents found."
            
            # Format the context
            context = self._format_context(docs)
            
            # Create the prompt
            prompt = f"""You are a knowledgeable IT support assistant helping users with technical issues. 
            
            Available context (from KB articles):
            {context}
            
            User's question: {question}
            
            Please provide a helpful and comprehensive response following these guidelines:
            1. First, try to answer based on the provided context
            2. If the context is incomplete but you can infer a likely solution, mention it
            3. For technical issues, suggest common troubleshooting steps if you can infer similar context
            4. If referring to specific documents or systems, include their links at the end
            5. Be concise but thorough in your explanation
            6. If the context is completely irrelevant, use your general knowledge to help
            
            Format your response in clear, easy-to-follow steps.
            """
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Print sources
            print("\nSources:")
            for i, doc in enumerate(docs, 1):
                print(f"{i}. {doc.metadata['source']}")
            
            return response.text
            
        except Exception as e:
            return f"Error querying the knowledge base: {str(e)}"

def main():
    # Initialize the query system
    try:
        kb_query = KBQueryGemini()
        print("Knowledge Base Query System (Powered by Gemini)")
        print("Type 'exit' to quit\n")
        
        # Interactive query loop
        while True:
            question = input("\nEnter your question: ").strip()
            if question.lower() in ('exit', 'quit'):
                break
                
            if not question:
                continue
                
            # Get and print the answer
            answer = kb_query.query(question)
            print("\nAnswer:")
            print(answer)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
