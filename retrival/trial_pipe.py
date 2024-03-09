import os
from dotenv import load_dotenv

load_dotenv('token.env')
api_token = os.getenv('api_token')

# from langchain.document_loaders import PyPDFLoader

# loader = PyPDFLoader("Accelerating-Apache-Spark-3.pdf")
# data = loader.load()

from langchain.document_loaders import TextLoader

loader = TextLoader("instructions.txt")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100
# )

# data = splitter.split_documents(data)
# len(data)

# r_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=450,
#     chunk_overlap=0, 
#     separators=["\n\n"]
# )

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=10, chunk_overlap=1)

# text_splitter = RecursiveCharacterTextSplitter(separators=["forest"], chunk_size=450, chunk_overlap=0)
docs = text_splitter.split_documents(data)

dash_line = "-" * 100
print(len(docs))
for doc in docs:
    print(doc.page_content)
    print(dash_line + '\n' + dash_line)

from langchain.embeddings import HuggingFaceEmbeddings

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

from langchain.vectorstores import Chroma


vectordb = Chroma.from_documents(
    documents=docs,
    embedding = embeddings,
    persist_directory = 'instructions'
    )

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "google/gemma-2b-it"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_name,token = api_token)
model = AutoModelForCausalLM.from_pretrained(model_name,token = api_token, device_map = 'auto', torch_dtype = dtype)

from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

pipe = pipeline(
 "text-generation", 
 model=model, 
 tokenizer=tokenizer,
 return_tensors='pt',
 max_new_tokens=100,
 model_kwargs={"torch_dtype": torch.bfloat16}
)

llm = HuggingFacePipeline(
 pipeline=pipe,
 model_kwargs={"temperature": 0.7},
)

qa = RetrievalQA.from_chain_type(
 llm=llm,
 chain_type="stuff",
 retriever=vectordb.as_retriever(search_kwargs={"k":3}),

)

print(qa.invoke("You are in enchanted woods. What can happen?"))