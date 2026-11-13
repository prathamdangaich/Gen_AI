import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


model = ChatNVIDIA(
#   model="nvidia/nemotron-3.5-lightning-30b-a3b",
  model = "nvidia/nemotron-3.5-bolt-e4b",
  api_key=os.getenv("NVIDIA_API_KEY"),
  max_completion_tokens=128,
)

template1 = PromptTemplate(
    template="Write a detailed report about {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary of the following text: {text}",
    input_variables=["text"]
)

# Without StrOutputParser

prompt1 = template1.invoke({"topic": "India"})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text":result1.content})
result2 = model.invoke(prompt2)

print(result2.content)
print()


# With StrOutputParser

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "USA"})
print(result)

# using chain.stream
# In stream we get the output word by word instead of whole answer at once.
# flush=True forces the output to appear on your screen immediately, rather than waiting in memory (a buffer).
parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

for chunk in chain.stream({"topic": "USA"}):
    print(chunk, end =" ", flush=True)