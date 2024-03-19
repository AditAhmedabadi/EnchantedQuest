# helper_functions.py

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os

# Load environment variables
load_dotenv('token.env')
api_token = os.getenv('api_token')

def load_document(file_path):
    """Load the document using TextLoader."""
    loader = TextLoader(file_path)
    return loader.load()

def split_document(data):
    """Split the document into chunks."""
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=1)
    return text_splitter.split_documents(data)

def generate_embeddings(model_name, model_kwargs, encode_kwargs):
    """Generate embeddings using HuggingFaceEmbeddings."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def create_vector_db(docs, embeddings, persist_directory):
    """Create a vector database using Chroma."""
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

def create_pipeline(model_name, api_token, dtype):
    """Create a pipeline for text generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=api_token, device_map='auto', torch_dtype=dtype)
    return pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        return_tensors='pt',
        max_new_tokens=40,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

def create_retrieval_qa(llm, chain_type, retriever):
    """Create a RetrievalQA instance."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
    )