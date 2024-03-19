from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import torch

model_kwargs = {
    'temperature': 0,
    'max_length' : 1000
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# model_device = torch.device()
model_id = 'google/gemma-2b'
llm = HuggingFacePipeline.from_model_id(
    model_id = model_id,
    task = 'text-generation',
    model_kwargs = model_kwargs,
    device = 'cuda'
)lsc