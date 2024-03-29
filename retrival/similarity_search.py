from dotenv import load_dotenv
import os
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from .helper_functions import load_document, split_document, generate_embeddings, create_vector_db, create_pipeline, create_retrieval_qa
from huggingface_hub import login

def main():
    # Load environment variables
    load_dotenv('token.env')
    api_token = os.getenv('api_token')

    login(api_token)

    # Load and split the document
    data = load_document("retrival/instructions.txt")
    docs = split_document(data)

    embeddings = generate_embeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': False}
    )
