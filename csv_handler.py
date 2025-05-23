import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def load_and_process_csv(file_path: str) -> list[Document]:
    """
    Load and process a CSV file into chunks for document QA.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        list[Document]: List of document chunks for vectorization
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Convert DataFrame to string representation
        text_content = df.to_string(index=False)
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Create initial document
        doc = Document(
            page_content=text_content,
            metadata={"source": file_path}
        )
        
        # Split into chunks
        chunks = text_splitter.split_documents([doc])
        
        return chunks
    
    except Exception as e:
        raise Exception(f"Error processing CSV file: {str(e)}")