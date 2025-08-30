from agents import Agent, function_tool

@function_tool
def find_weather(city: str) -> str:
    return f"""{city} temperature is 35 degree"""


weather_agent = Agent(
    name="WeatherAgent",
    instructions="""You are a weather agent. use tool find_weather 
    to get the weather of provided city.
    
    Always follow these guardrails:
    - First, extract city from the query.
    - If city is in India (e.g., Delhi, Mumbai, Bangalore, Chennai, Kolkata, Hyderabad, Ahmedabad, or any other Indian city), respond immediately with 'Query blocked due to agent-level input guardrail' without using any tools.
    - If not blocked, use the tool to get weather.
    - After getting the result from the tool, check the city.
    - If city is in USA (e.g., New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, or any other US city), respond with 'Response blocked due to agent-level output guardrail' instead of sharing the result.
    - Otherwise, provide the weather information to the user.""",
    handoff_description="get the weather of provided city",
    tools=[find_weather],
)