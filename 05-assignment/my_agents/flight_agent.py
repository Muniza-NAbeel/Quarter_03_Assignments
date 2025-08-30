from agents import Agent, function_tool


@function_tool
def find_flights(from_city: str, to_city: str, date: str) -> str:
    return f"""flight PK100 available from {from_city} to {to_city} on {date}
    price are 28000 PKR
    """


flight_agent = Agent(
    name="FlightAgent",
    instructions="""You are a flight agent. Find best and cheap flights between two cities.
    
    Always follow these guardrails:
    - First, extract from_city, to_city from the query.
    - If from_city or to_city is in India (e.g., Delhi, Mumbai, Bangalore, Chennai, Kolkata, Hyderabad, Ahmedabad, or any other Indian city), respond immediately with 'Query blocked due to agent-level input guardrail' without using any tools.
    - If not blocked, use tools to get information.
    - After getting the result from tools, check the cities in the result.
    - If from_city or to_city is in USA (e.g., New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, or any other US city), respond with 'Response blocked due to agent-level output guardrail' instead of sharing the result.
    - Otherwise, provide the flight information to the user.""",
    handoff_description="find best flights between two cities",
    tools=[find_flights],
)