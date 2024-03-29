from dotenv import load_dotenv
import os
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from .helper_functions import load_document, split_document, generate_embeddings, create_vector_db, create_pipeline, create_retrieval_qa
from huggingface_hub import login

# main.py

def main():
    # Load environment variables
    load_dotenv('token.env')
    api_token = os.getenv('api_token')

    login(api_token)

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
    pipe = create_pipeline("google/gemma-2b-it", api_token, torch.bfloat16)

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

    start_prompt = '''GAME DIALOGUE:
model:
The beast lunges at you, its claws outstretched. You narrowly avoid its attack and find yourself in a standoff, your strength against its fury. You know that you must find a way to defeat this beast, but you are running out of time. What will you do next?<end_of_turn>
user:
i take my axe and cut tis head off<end_of_turn>

based on the dialogue above, give relevant instructions to model to continue the story.'''


    result = qa.invoke(start_prompt)
    return result

if __name__ == "__main__":
    print(main())
