import os 
import logging 
from typing import List 
from azure.search.documents import SearchClient 
from azure.search.documents.models import VectorizedQuery 
from azure.core.credentials import AzureKeyCredential 
from chunks import vectorize  
from openai import AzureOpenAI 
# import numpy as np 
import os 
import fitz 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.schema import Document 
from dotenv import load_dotenv 
from pathlib import Path 
import uuid
import hashlib

load_dotenv()  

endpoint = os.getenv("AZURE_SEARCH_ENDPOINT") 
api_key = os.getenv("AZURE_SEARCH_API_KEY") 
index = os.getenv("INDEX")   

credential=AzureKeyCredential(api_key)  

search_client = SearchClient(     
    endpoint=endpoint,      
    index_name=index,      
    credential=AzureKeyCredential(api_key)     
)  

def extract_txt(document):     
    doc = fitz.open(document)     
    text = ""     
    for page in doc:         
        text += page.get_text()     
    doc.close()     
    return text  

def chunks(text):     
    doc = Document(page_content=text)     
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(         
        chunk_size=500,         
        chunk_overlap=50         
    )     
    chunk = text_splitter.split_documents([doc])      
    return chunk  

def vectorize(chunk):     
    client = AzureOpenAI(         
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),           
        api_version= os.getenv("AZURE_OPENAI_API_VERSION"),         
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")     
    )     
    response = client.embeddings.create(         
        input = str(chunk),         
        model = "text-embedding-3-small"     
    )     
    result_embedding = response.data[0].embedding      
    return result_embedding  

def sanitize_filename(filename):
    """
    Sanitize filename to be Azure Search key compliant
    Azure Search keys can only contain: letters, digits, underscore (_), dash (-), or equal sign (=)
    """
    import re
    name = Path(filename).stem
    sanitized = re.sub(r'[^a-zA-Z0-9_\-=]', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"doc_{sanitized}" 
    return sanitized

def generate_unique_id(file_path, chunk_index, method="filename_index"):
    """
    Generate unique ID using different methods - Azure Search key compliant
    
    Args:
        file_path: Path to the PDF file
        chunk_index: Index of the chunk within the document
        method: Method to use for ID generation
    
    Returns:
        Unique string ID (Azure Search compliant)
    """
    file_name = sanitize_filename(file_path)  # Sanitize the filename
    
    if method == "filename_index":
        return f"{file_name}_chunk_{chunk_index:04d}"
    elif method == "uuid":
        return str(uuid.uuid4()).replace('-', '_')
    elif method == "hash_based":
        content = f"{file_name}_{chunk_index}"
        return f"hash_{hashlib.md5(content.encode()).hexdigest()[:16]}"
    elif method == "timestamp_based":
        import time
        timestamp = int(time.time())
        return f"{file_name}_{timestamp}_{chunk_index:04d}"
    elif method == "simple_counter":
        return f"doc_{abs(hash(str(file_path)))%100000:05d}_{chunk_index:04d}"
    else:
        return f"{file_name}_chunk_{chunk_index:04d}"

def document_upload(chunks_list, file_path, id_method="filename_index"):     
    """
    Upload document chunks with unique IDs
    
    Args:
        chunks_list: List of document chunks
        file_path: Path to the source PDF file
        id_method: Method to use for generating unique IDs
    """
    uploaded_count = 0
    
    for chunk_index, chunk in enumerate(chunks_list, 1):
        
        unique_id = generate_unique_id(file_path, chunk_index, method=id_method)
        
        document = {             
            "id": unique_id,  
            "chunks": chunk.page_content,             
            "embeddings": vectorize(chunk.page_content),             
            "category": ["drugs","decisions"]  
        }
        
        try:
            result = search_client.upload_documents(documents=[document])
            uploaded_count += 1
            print(f"File: {Path(file_path).name} | Chunk: {chunk_index} | ID: {unique_id} | Status: Success")
        except Exception as e:
            print(f"Error uploading chunk {chunk_index} from {Path(file_path).name}: {e}")
    
    print(f"Upload complete for {Path(file_path).name}! Uploaded {uploaded_count} chunks.")
    return uploaded_count

def upload_documents_From_folder():          
    folder_path = Path("Your folder path")     
    pdf_files = list(folder_path.glob("*.pdf"))
    
    total_uploaded = 0
    processed_files = 0
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    for pdf_index, pdf in enumerate(pdf_files, 1):         
        print(f"\n{'='*50}")
        print(f"Processing file {pdf_index}/{len(pdf_files)}: {pdf.name}")
        print(f"{'='*50}")
        
        try:
            text = extract_txt(pdf)
            if not text.strip():
                print(f"Warning: No text extracted from {pdf.name}")
                continue
                
            chunk_list = chunks(text)
            print(f"Created {len(chunk_list)} chunks from {pdf.name}")
        
            uploaded = document_upload(chunk_list, pdf, id_method="filename_index")
            
            total_uploaded += uploaded
            processed_files += 1
            
        except Exception as e:
            print(f"Error processing {pdf.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"UPLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {processed_files}/{len(pdf_files)}")
    print(f"Total chunks uploaded: {total_uploaded}")
    print(f"Upload complete!")

class DocumentUploader:
    def __init__(self):
        self.global_chunk_counter = 0
    
    def generate_global_unique_id(self, file_path):
        """Generate ID with global counter - Azure Search compliant"""
        self.global_chunk_counter += 1
        file_name = sanitize_filename(file_path)
        return f"{file_name}_{self.global_chunk_counter:06d}"
    
    def upload_with_global_counter(self, chunks_list, file_path):
        """Upload using global counter for IDs"""
        uploaded_count = 0
        
        for chunk in chunks_list:
            unique_id = self.generate_global_unique_id(file_path)
            
            document = {
                "id": unique_id,
                "chunks": chunk.page_content,
                "embeddings": vectorize(chunk.page_content),
                "doc_category": "familyCode_decisions"
            }
            
            try:
                result = search_client.upload_documents(documents=[document])
                uploaded_count += 1
                print(f"Global ID: {unique_id} | Status: Success")
            except Exception as e:
                print(f"Error uploading {unique_id}: {e}")
        
        return uploaded_count

if __name__ == "__main__":    
    upload_documents_From_folder()    
