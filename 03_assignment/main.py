import os
from agents import Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from my_agents.agent import human_agent, bot_agent
from agents import set_tracing_disabled

# Disable LiteralAI/OpenAI instrumentation
os.environ["LITERALAI_DISABLED"] = "1"
set_tracing_disabled(True)  # disable internal tracing


def chat():
    print("🤖 Welcome! I'm your customer service assistant.")
    print("I can help with order status, product info, and more.")
    print("If I can't help, I'll transfer you to a human agent.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break

        try:
            # Try bot agent first
            result = Runner.run_sync(
                starting_agent=bot_agent,
                input=user_input,
                context={"user_input": user_input}
            )
            print(f"🤖 Bot: {result.final_output}")
        except InputGuardrailTripwireTriggered:
            # If guardrail triggered, use human agent
            result = Runner.run_sync(
                starting_agent=human_agent,
                input=user_input,
                context={"user_input": user_input}
            )
            print(f"👤 Human Agent: {result.final_output}")


if __name__ == "__main__":
    chat()
