import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class KBQuery:
    def __init__(self, persist_dir: str = "faiss_index"):
        """Initialize the KB query system"""
        self.persist_dir = Path(persist_dir)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectorstore = self._load_vectorstore()
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.qa_chain = self._create_qa_chain()

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

    def _create_qa_chain(self):
        """Create a QA chain with a custom prompt"""
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def query(self, question: str) -> str:
        """Query the knowledge base with a question"""
        print(f"\nSearching for: {question}")
        
        try:
            result = self.qa_chain({"query": question})
            
            # Print the sources
            print("\nSources:")
            for i, doc in enumerate(result['source_documents']):
                print(f"{i+1}. {doc.metadata['source']}")
            
            return result['result']
            
        except Exception as e:
            return f"Error querying the knowledge base: {str(e)}"

def main():
    # Initialize the query system
    try:
        kb_query = KBQuery()
        print("Knowledge Base Query System")
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
