import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.tools import tool

# Load environment variables from .env file
load_dotenv()

# Configure OpenRouter model
# IMPORTANT: API key is loaded from .env file (see .env.example)
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1000,
)

# Example tool: Temperature conversion
@tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert temperature between Celsius and Fahrenheit.
    
    Args:
        value: The temperature value to convert
        from_unit: Source unit ("celsius" or "fahrenheit")
        to_unit: Target unit ("celsius" or "fahrenheit")
    
    Returns:
        Converted temperature value
    """
    if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        return (value * 9/5) + 32
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        return (value - 32) * 5/9
    elif from_unit.lower() == to_unit.lower():
        return value
    else:
        raise ValueError(f"Unsupported conversion from {from_unit} to {to_unit}")
        
#agent creation
agent = create_agent(
    model=model,
    system_prompt="Always use the convert_temperature tool when users ask for temperature conversions.",
    tools=[convert_temperature]
)

# Test 1: Simple conversion
response = agent.invoke({"messages": [HumanMessage("Convert 25 degrees Celsius to Fahrenheit")]})
print(response["messages"][-1].content)

# Test 2: Reverse conversion
response = agent.invoke({"messages": [HumanMessage("What is 77 degrees Fahrenheit in Celsius?")]})
print(response["messages"][-1].content)

# Test 3: More complex question
response = agent.invoke({"messages": [HumanMessage("If it's 20°C outside, what would that be in Fahrenheit? Is that warm or cold?")]})
print(response["messages"][-1].content)
