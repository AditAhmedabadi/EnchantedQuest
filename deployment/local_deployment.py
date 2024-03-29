from langchain_community.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import torch
from langchain.chains import LLMChain

model_kwargs = {
    'temperature': 0,
    'max_length' : 100
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# model_device = torch.device()
model_id = 'google/gemma-7b-it'
llm = HuggingFacePipeline.from_model_id(
    model_id = model_id,
    task = 'text-generation',
    model_kwargs = model_kwargs,
    # device = 'auto'
)

template = """
You are a friendly chatbot assistant that responds conversationally to users' questions.
Keep the answers short, unless specifically asked by the user to elaborate on something.

Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

def ask_question(question):
    result = llm_chain(question)
    print(result['question'])
    print("")
    print(result['text'])

ask_question("Describe vegeta's personality development")