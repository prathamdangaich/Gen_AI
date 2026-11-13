import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()


model = ChatNVIDIA(
#   model="nvidia/nemotron-3.5-lightning-30b-a3b",
  model = "nvidia/nemotron-3.5-bolt-e4b",
  api_key=os.getenv("NVIDIA_API_KEY"),
  max_completion_tokens=128,
)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Generate a name with age, nationality and sex. \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}  #partial_variables is used to load variable at creation time
)


# Without using chains

prompt= template.format()

result = model.invoke(prompt)
print(result.content)
print(type(result))

print()

final_result = parser.parse(result.content)
print(final_result)
print(type(final_result))


# By using chains

chain = template | model | parser
result = chain.invoke({})
print(result)