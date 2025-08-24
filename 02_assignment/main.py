# hotel_assistant.py
from typing import Any
from agents import (
    Agent,
    Runner,
    RunContextWrapper,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    input_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
)
from decouple import config
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AsyncOpenAI

load_dotenv()
set_tracing_disabled(True)

key = config("GEMINI_API_KEY")
base_url = config("GEMINI_BASE_PATH")

gemini_client = AsyncOpenAI(api_key=str(key), base_url=str(base_url))

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=gemini_client,
)

# Example hotel data
hotels_data = {
    "Grand Palace": {
        "location": "Karachi, Pakistan",
        "rooms": "Luxury suites, standard rooms",
        "price": "Rs. 20,000 per night",
        "contact": "+92-310-786543"
    },
    "Sea View Hotel": {
        "location": "Karachi Beachfront",
        "rooms": "Sea view deluxe rooms",
        "price": "Rs. 15,000 per night",
        "contact": "+92-310-7654321"
    }
}

dynamic_instructions = (
    "You are a helpful hotel booking assistant. "
    "Ask the user which hotel they want details for. "
    "If the user asks for details about a Grand Palace or Sea View they are asking about Grand Palace hotel or Sea View hotel, "
    "Available hotels: " + ", ".join(hotels_data.keys())
)

class MyDataType(BaseModel):
    is_query_about_Grand_Palace_Hotel_or_Sea_View_Hotel: bool
    reason: str

guardrial_agent = Agent(
    name="GuardrailAgent",
    instructions=(
        "You are a query classifier. Determine if the user's query is about Grand Palace Hotel or Sea View Hotel. "
        "Return is_query_about_Grand_Palace_Hotel_or_Sea_View_Hotel = true if the query mentions: "
        "- Grand Palace (hotel) "
        "- Sea View Hotel "
        "- Hotel booking, accommodation, rooms, pricing for these hotels "
        "Return is_query_about_Grand_Palace_Hotel_or_Sea_View_Hotel = false for any other topics like weather, politics, general questions, etc."
    ),
    model=gemini_model,
    output_type=MyDataType
)

@input_guardrail
async def guardrial_input_function(ctx: RunContextWrapper, agent, input):
    result = await Runner.run(guardrial_agent, input=input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_query_about_Grand_Palace_Hotel_or_Sea_View_Hotel
    )

agent = Agent(
    name="HotelAssistant",
    instructions=dynamic_instructions,
    model=gemini_model,
    input_guardrails=[guardrial_input_function]
)


async def main():
    try:
        while True:
            msg = input("Enter your question (or 'exit' to quit): ")
            if msg.lower() == "exit":
                print("Exiting Hotel Assistant...")
                break

            try:
                res = await Runner.run(
                    starting_agent=agent,
                    input=msg,
                )
                print(f"\nResponse: {res.final_output}\n")
            except InputGuardrailTripwireTriggered as e:
                print("❌ I can only help with hotel-related queries! Ask me about Grand Palace or Sea View hotels.\n")
    except KeyboardInterrupt:
        print("\nProgram terminated manually.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
