import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema #depricated so, moved to classic
from dotenv import load_dotenv

load_dotenv()


model = ChatNVIDIA(
#   model="nvidia/nemotron-3.5-lightning-30b-a3b",
  model = "nvidia/nemotron-3.5-bolt-e4b",
  api_key=os.getenv("NVIDIA_API_KEY"),
  max_completion_tokens=128,
)

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 facts about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# Clean LCEL Pipeline:
chain = template | model | parser

result = chain.invoke({"topic": "Black Hole"})
print(result)