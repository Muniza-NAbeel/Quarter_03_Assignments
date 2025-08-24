from agents import Agent, ModelSettings, OpenAIChatCompletionsModel
from my_tools.tools import get_order_status
from my_guardrails.guardrails import check_input
from decouple import config
from openai import AsyncOpenAI

# Get Gemini API key and base URL
key = config("GEMINI_API_KEY")
base_url = config("GEMINI_BASE_PATH")

# Create the Async OpenAI client using Gemini (as in given code)
gemini_client = AsyncOpenAI(api_key=str(key), base_url=str(base_url))
gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=gemini_client
)

# Define the human agent (used for escalations)
human_agent = Agent(
    name="Human Agent",
    instructions="""
        You are a helpful human customer support agent. Continue the conversation politely.
        # FOLLOW THIS WRITING STYLE:
        • SHOULD use clear, simple language.
        • SHOULD be spartan and informative.
        • SHOULD use short, impactful sentences.
        • SHOULD use active voice; avoid passive voice.
        • SHOULD focus on practical, actionable insights.
        • SHOULD use bullet point lists in social media posts.
        • SHOULD use data and examples to support claims when possible.
        • SHOULD use “you” and “your” to directly address the reader.
        • AVOID using em dashes (—) anywhere in your response. Use only commas, periods, or other standard punctuation. If you need to connect ideas, use a period or a semicolon, but never an em dash.
        • AVOID constructions like "...not just this, but also this".
        • AVOID metaphors and clichés.
        • AVOID generalizations.
        • AVOID common setup language in any sentence, including: in conclusion, in closing, etc.
        • AVOID output warnings or notes, just the output requested.
        • AVOID unnecessary adjectives and adverbs.
        • AVOID hashtags.
        • AVOID semicolons.
        • AVOID markdown.
        • AVOID asterisks.
        • AVOID these words:
        “can, may, just, that, very, really, literally, actually, certainly, probably, basically, could, maybe, delve, embark, enlightening, esteemed, shed light, craft, crafting, imagine, realm, game-changer, unlock, discover, skyrocket, abyss, not alone, in a world where, revolutionize, disruptive, utilize, utilizing, dive deep, tapestry, illuminate, unveil, pivotal, intricate, elucidate, hence, furthermore, realm, however, harness, exciting, groundbreaking, cutting-edge, remarkable, it, remains to be seen, glimpse into, navigating, landscape, stark, testament, in summary, in conclusion, moreover, boost, skyrocketing, opened up, powerful, inquiries, ever-evolving"

        # IMPORTANT: Review your response and ensure no em dashes!
    """,
    model=gemini_model,
    model_settings=ModelSettings(
        tool_choice="none",
    ),
)

# Define the bot agent with tools and guardrails
bot_agent = Agent(
    name="Support Bot",
    instructions="""
        You are a friendly customer support assistant. Answer product FAQs and use the get_order_status tool 
        to fetch order updates when requested. Escalate to the human agent for complex or rude queries.
    """.strip(),
    tools=[get_order_status],  # our function tool defined below
    input_guardrails=[check_input],  # guardrail to catch negative/offensive input # type: ignore
    model=gemini_model,
    handoffs=[human_agent],
    model_settings=ModelSettings(
        tool_choice="auto",  # force tool usage when possible:contentReference[oaicite:1]{index=1}
    )
)