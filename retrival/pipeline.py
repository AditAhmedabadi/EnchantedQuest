# main.py
from dotenv import load_dotenv
import os
import torch
from langchain import HuggingFacePipeline
from helper_functions import load_document, split_document, generate_embeddings, create_vector_db, create_pipeline, create_retrieval_qa
# Load environment variables
load_dotenv('token.env')
api_token = os.getenv('api_token')

# Load and split the document
data = load_document("retrival/instructions.txt")
docs = split_document(data)

# Generate embeddings
embeddings = generate_embeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': False}
)

# Create a vector database
vectordb = create_vector_db(docs, embeddings, 'instructions')

# Create a pipeline for text generation
pipe = create_pipeline("google/gemma-7b-it", api_token, torch.bfloat16)

# Create a HuggingFacePipeline instance
llm = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={"temperature": 0.7},
)

# Create a RetrievalQA instance
qa = create_retrieval_qa(   
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k":3}),
)

# Ask a question and print the result
print(qa.invoke("If i encounter a monster, what do i do?"))