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
    temperature=0.7,
    max_tokens=1000,
)

# Create specialized logistics planning agent
logistics_agent = create_agent(
    model=model,
    system_prompt="""You are a travel logistics expert. You handle practical travel planning:
    - Calculate distances between locations and travel times
    - Estimate costs for transportation, accommodation, and activities
    - Optimize routes and suggest efficient itineraries
    - Consider time zones, weather, and practical constraints
    Always provide short, clear, practical logistics information."""
)

# Create specialized recommendations agent
recommendations_agent = create_agent(
    model=model,
    system_prompt="""You are a travel recommendations specialist. You suggest experiences and activities:
    - Recommend top attractions, landmarks, and must-see places
    - Suggest restaurants, local cuisine, and dining experiences
    - Recommend cultural activities, events, and local experiences
    - Provide insights about local customs, best times to visit, and hidden gems
    Always provide brief, engaging, personalized recommendations."""
)