import os
import json
import requests
from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-70B-Instruct",
    task="text-generation",
    max_new_tokens=512,       # FIX 1: Prevent truncated/incomplete responses
    temperature=0.1,          # FIX 2: Lower temp = more deterministic tool use
)
model = ChatHuggingFace(llm=llm)

@tool
def get_weather(city: str) -> str:
    """Fetches the current weather for a given city. Returns a human-readable summary."""
    weather_url = (
        f"http://api.weatherstack.com/current"
        f"?access_key={os.getenv('WEATHER_API_KEY')}"
        f"&query={city}"
    )
    response = requests.get(weather_url)
    
    if response.status_code != 200:
        return "Unable to fetch weather data."
    
    data = response.json()
    
    # FIX 3: Return a clean string, not raw JSON
    # Raw JSON confuses the LLM and causes it to loop trying to "parse" it
    try:
        current = data["current"]
        location = data["location"]
        return (
            f"Weather in {location['name']}, {location['country']}: "
            f"{current['weather_descriptions'][0]}, "
            f"Temperature: {current['temperature']}°C, "
            f"Feels like: {current['feelslike']}°C, "
            f"Humidity: {current['humidity']}%, "
            f"Wind: {current['wind_speed']} km/h"
        )
    except (KeyError, IndexError):
        return f"Received weather data but couldn't parse it: {data}"

search_tool = DuckDuckGoSearchRun()
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=model, tools=[get_weather, search_tool], prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[get_weather, search_tool],
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=5,
    max_execution_time=30,
    # early_stopping_method="generate"  ← remove this line
)

response = agent_executor.invoke({"input": "What is the capital of Israel and what is the weather there?"})
print(response['output'])