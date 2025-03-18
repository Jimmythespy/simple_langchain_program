from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub

from pydantic import BaseModel

import os
import shutil
import aiofiles

app = FastAPI() # Initialize FastAPI app
templates = Jinja2Templates(directory="templates") # Set up Jinja2 template directory

# Load environment variables 
os.environ["OPENAI_API_KEY"] = ""

# Load and Split Documents
def load_and_preprocess_documents(file_path):
    loader = PyPDFLoader(file_path)  # Load PDF
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)  # Split into chunks
    return docs

# Generate FAISS Index and Save It
def create_and_save_faiss_index(file_path, index_path="vector_store"):
    print("Creating new index!")
    documents = load_and_preprocess_documents(file_path)
    embeddings = OpenAIEmbeddings()  
    
    vector_store = FAISS.from_documents(documents, embeddings)  # Create FAISS index

    # Save FAISS index to disk
    vector_store.save_local(index_path)
    print(f"FAISS index saved at: {index_path}")

# Initialize embedding model
embedding_model = OpenAIEmbeddings()

# Try to load FAISS index from local
try:
    vector_store = FAISS.load_local("vector_store", embedding_model, allow_dangerous_deserialization=True)
    print("index already exsited!")
except:
    # Create an new FAISS file
    file = os.listdir("uploads/")[0]
    file_path = f"uploads/{file}"  
    create_and_save_faiss_index(file_path)
    vector_store = FAISS.load_local("vector_store", embedding_model, allow_dangerous_deserialization=True)
    print("Creating new index!")

# Create a retrieval-based QA chain
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Query stucture
class Query(BaseModel):
    question: str
    time: int | None = None

# FastAPI 
@app.get("/query_ui", response_class=HTMLResponse)
async def query_form(request: Request):
    return templates.TemplateResponse("query.html", {"request": request})

# Serve HTML page for file upload
@app.get("/uploadfile_ui", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# Return the list of file in the vector store
@app.get("/listfile")
async def list_file(): 
    file = os.listdir("uploads/")
    return {"file_list" : file}

# Endpoint to handle file upload
@app.post("/uploadfile")
async def upload_file(file: UploadFile = File(...)):
    global vector_store
    global embedding_model
    
    # Define the location where the file will be saved
    file_location = f"uploads/{file.filename}"
    
    # Use aiofiles to handle async file writing
    async with aiofiles.open(file_location, 'wb') as f:
        content = await file.read()  # Read file content
        await f.write(content)       # Write the content to the file
    
    # Add the downloaded document to the vector store
    docs = load_and_preprocess_documents(file_location) # Load the newly downloaded document
    
    try:
        await vector_store.aadd_documents(docs)
        
        # Store the FAISS index to local
        vector_store.save_local("vector_store")
    except Exception as e: 
        print(e)
        raise HTTPException(status_code=500, detail=str(e)) 
    
    # return RedirectResponse(url="/query_ui", status_code=status.HTTP_303_SEE_OTHER)
    return {"filename": file_location}
    # except Exception as e:
    #     print(e)
    #     raise HTTPException(status_code=400, detail=str(e)) 

# Endpoint to handle query
@app.post("/query")
async def query_vector_db(query: Query): 
    try:
        answer = qa_chain.invoke({"input": query.question})
        return {"question": query.question, "answer": answer['answer']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))