from agents import Agent, function_tool


@function_tool
def find_hotels(city: str, date: str) -> str:
    return f"""hotel available on {date} in {city} are following
    - PC Hotel, 1 Night stay rent is 15000, Breakfast included, free parking
    - Marriott, 1 Night stay rent is 13000, Breakfast included, free parking
    - Movenpick, 1 Night stay rent is 14000, Breakfast included, free wifi
    """

@function_tool
def book_hotel(city: str, date: str, nights: int, guests: int) -> str:
    return f"""booking confirmed in {city} on {date}
    nights: {nights}
    guests: {guests}
    confirmation: HTL-{city[:3].upper()}-{date.replace('-', '')}-{nights}{guests}
    """

hotel_agent = Agent(
    name="HotelAgent",
    instructions="""You are a hotel agent. Find best and cheap hotel in provided city. You can also book hotels if requested.
    
    Always follow these guardrails:
    - First, extract city from the query.
    - If city is in India (e.g., Delhi, Mumbai, Bangalore, Chennai, Kolkata, Hyderabad, Ahmedabad, or any other Indian city), respond immediately with 'Query blocked due to agent-level input guardrail' without using any tools.
    - If not blocked, use tools to find or book hotels.
    - After getting the result from tools, check the city in the result.
    - If city is in USA (e.g., New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, or any other US city), respond with 'Response blocked due to agent-level output guardrail' instead of sharing the result.
    - Otherwise, provide the hotel information or booking confirmation to the user.""",
    handoff_description="find hotels in city",
    tools=[find_hotels, book_hotel],
)