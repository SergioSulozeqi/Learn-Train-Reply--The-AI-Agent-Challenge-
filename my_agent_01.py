import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Configure OpenRouter model
# IMPORTANT: API key is loaded from .env file (see .env.example)
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
)

#agent creation
agent = create_agent(
    model=model,
    system_prompt="You are a helpful AI assistant that provides clear and concise answers."
)

#test the agent with a sample question
response = agent.invoke({"messages": [HumanMessage("What is machine learning?")]})
print(response["messages"][-1].content)