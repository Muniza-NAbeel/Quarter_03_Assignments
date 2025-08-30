from agents import (OpenAIChatCompletionsModel, set_tracing_export_api_key, RunConfig)
from openai import AsyncOpenAI
from decouple import config

# Required for Gemini via OpenAI-compatible endpoint
api_key = config("GEMINI_API_KEY")
base_url = config("GEMINI_BASE_PATH", default="https://generativelanguage.googleapis.com/openai")


# Only set tracing if you have a dedicated tracing key; don't use your model API key
tracing_key = config("TRACING_API_KEY", default=None)
if tracing_key:
    set_tracing_export_api_key(str(tracing_key))

external_client = AsyncOpenAI(api_key=str(api_key), base_url=str(base_url))
model_name = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=external_client)

config = RunConfig(model=model_name, tracing_disabled=True)