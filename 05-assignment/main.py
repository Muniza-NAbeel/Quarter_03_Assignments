from typing import Literal
from agents import (
    Agent,
    HandoffInputData,
    RunContextWrapper,
    Runner,
    RunConfig,
    TResponseInputItem,
    handoff,
    set_tracing_disabled,
    input_guardrail,
    output_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered
)
from pydantic import BaseModel
from my_agents.weather_agent import weather_agent
from my_agents.hotel_agent import hotel_agent
from my_agents.flight_agent import flight_agent
from my_config.config import model_name
from agents.extensions import handoff_filters
import asyncio

set_tracing_disabled(True)

class Users(BaseModel):
    name: str
    role: Literal["admin", "super user", "basic"]
    age: int


async def handoff_permission(ctx: RunContextWrapper[Users], agent: Agent) -> bool:
    if ctx.context.age > 25:
        return True

    if ctx.context.role == "super user":
        return True
    return False


def handoff_filter(data: HandoffInputData) -> HandoffInputData:
    data = handoff_filters.remove_all_tools(data)
    history = data.input_history[-2:]

    return HandoffInputData(
        input_history=history,
        new_items=data.new_items,
        pre_handoff_items=data.pre_handoff_items,
    )


triage_agent = Agent(
    name="TriageAgent",
    instructions="""
    you are a triage agent, hand off to flight,hotel or weather agent 
    if user ask for otherwise you can response yourself
    """,
    handoffs=[
        handoff(
            agent= weather_agent,
            tool_name_override= "handoff_weatheragent",
            tool_description_override= "handoff to weather agent to get the weather information",
            is_enabled= handoff_permission,
            input_filter= handoff_filter,
        ),
        hotel_agent,
        flight_agent,
    ],
    handoff_description="""
    this triage agent, hand off to flight,hotel or weather agent 
    if user ask for otherwise you can response yourself""",
)

indian_cities = [
    "delhi", "new delhi", "mumbai", "bombay", "bangalore", "bengaluru",
    "chennai", "madras", "kolkata", "calcutta", "hyderabad", "ahmedabad",
    "pune", "jaipur", "lucknow", "kanpur", "nagpur", "indore", "bhopal",
    "patna", "vadodara", "ludhiana", "agra", "nashik", "surat", "coimbatore"
]
us_cities = [
    "new york", "los angeles", "chicago", "houston", "phoenix", 
    "philadelphia", "san antonio", "san diego", "dallas", "san jose",
    "austin", "jacksonville", "fort worth", "columbus", "charlotte", 
    "san francisco", "indianapolis", "seattle", "denver", "washington"
]

def has_indian_city(text: str) -> bool:
    return any(city in text.lower() for city in indian_cities)

def has_us_city(text: str) -> bool:
    return any(city in text.lower() for city in us_cities)

@input_guardrail
async def agent_input_guardrail(ctx: RunContextWrapper, agent: Agent, input_data: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    if isinstance(input_data, list):
        query = input_data[-1].get('content', '') if input_data else ''
    else:
        query = input_data
    triggered = has_indian_city(query)
    return GuardrailFunctionOutput(
        tripwire_triggered=triggered,
        output_info={"reason": "Contains Indian city"} if triggered else {"reason": "OK"}
    )

@output_guardrail
async def agent_output_guardrail(ctx: RunContextWrapper, agent: Agent, output: str) -> GuardrailFunctionOutput:
    triggered = has_us_city(output)
    return GuardrailFunctionOutput(
        tripwire_triggered=triggered,
        output_info={"reason": "Contains US city"} if triggered else {"reason": "OK"}
    )

# Add guardrails to the agents
weather_agent.input_guardrails = [agent_input_guardrail]
weather_agent.output_guardrails = [agent_output_guardrail]
hotel_agent.input_guardrails = [agent_input_guardrail]
hotel_agent.output_guardrails = [agent_output_guardrail]
flight_agent.input_guardrails = [agent_input_guardrail]
flight_agent.output_guardrails = [agent_output_guardrail]

weather_agent.handoffs.append(triage_agent)
hotel_agent.handoffs.append(triage_agent)
flight_agent.handoffs.append(triage_agent)


async def main():
    user = Users(name="abc", role="super user", age=20)
    start_agent = triage_agent
    input_data: list[TResponseInputItem] = []
    while True:
        user_prompt = input("enter your query : ")
        if user_prompt == "exit":
            break

        input_data.append({"role": "user", "content": user_prompt})

        try:
            result = await Runner.run(
                start_agent,
                input=input_data,
                run_config=RunConfig(model= model_name),
                context=user,
            )
            start_agent = result.last_agent
            input_data = result.to_input_list()

            # Run-level output guardrail (manual)
            final_output = result.final_output
            if has_us_city(final_output):
                print("❌ No output due to run-level output guardrail")
            else:
                print(final_output)
        except InputGuardrailTripwireTriggered:
            print("❌ Reject query")
            # Reset input_data to previous if needed, but for simplicity, remove the last input
            input_data.pop()
        except OutputGuardrailTripwireTriggered:
            print("❌ No output due to agent-level output guardrail")
            # Similar, pop if needed
            input_data.pop()


asyncio.run(main())