import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv

load_dotenv()


model = ChatNVIDIA(
#   model="nvidia/nemotron-3.5-lightning-30b-a3b",
  model = "nvidia/nemotron-3.5-bolt-e4b",
  api_key=os.getenv("NVIDIA_API_KEY"),
  max_completion_tokens=128,
)

messages = [
    SystemMessage(content = "You are a helpful agent"),
    HumanMessage(content="Tell me about Langchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)
