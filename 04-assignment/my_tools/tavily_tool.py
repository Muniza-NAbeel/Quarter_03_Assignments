import json
import os
from typing import TypedDict
from dotenv import load_dotenv
from tavily import TavilyClient
from agents import function_tool  

load_dotenv()

class TavilyArgs(TypedDict):
    """Arguments for Tavily search."""
    query: str
    max_results: int | None
    include_answer: bool | None

@function_tool
def tavily_search(args: TavilyArgs) -> str:
    """
    Run a web search with Tavily and return a compact JSON summary.

    Args:
      query: What to search for on the web.
      max_results: Limit number of results to keep (default 5).
      include_answer: Ask Tavily to generate a direct answer when possible.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY missing")

    client = TavilyClient(api_key=api_key)

    # Tavily's client accepts a string (basic) or keyword params via **kwargs.
    # We'll stay defensive and only forward what we declared.
    q = args["query"]
    max_results = args.get("max_results") or 5
    include_answer = args.get("include_answer")
    resp = client.search(q)  # basic usage per quickstart

    # Normalize to a compact shape that's easy for the model:
    answer = resp.get("answer")
    results = resp.get("results", [])[:max_results]
    slim = {
        "query": resp.get("query", q),
        "answer": answer,
        "results": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": r.get("content"),
                "score": r.get("score"),
            }
            for r in results
        ],
    }
    # Ensure we always return a string (SDK expects str output for function tools)
    return json.dumps(slim, ensure_ascii=False)