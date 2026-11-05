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

# For storing history or context
chat_history = [
    SystemMessage(content="You are a helpful AI Agent")
] 

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input == "exit":
        print("Thank you for using")
        break
    
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)
print(chat_history)