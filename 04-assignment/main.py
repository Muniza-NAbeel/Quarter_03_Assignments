import sys
import traceback

def main():
    print("🚀 Starting application...")
    
    try:
        # Import section with individual error handling
        print("📦 Loading imports...")
        
        import json
        import os
        from dataclasses import dataclass
        from openai import AsyncOpenAI
        from dotenv import load_dotenv
        from pydantic import BaseModel
        from tavily import TavilyClient
        
        # This is the most likely problematic import
        try:
            from agents import (
                Agent, 
                Runner, 
                RunContextWrapper, 
                function_tool, 
                set_tracing_disabled,
                OpenAIChatCompletionsModel
            )
            print("✅ Agents import successful")
        except ImportError as e:
            print(f"❌ Failed to import 'agents' package: {e}")
            print("💡 Try installing: uv add openai-agents")
            sys.exit(1)
            
        # Load environment variables
        print("🔧 Loading environment variables...")
        load_dotenv()
        
        # Check required environment variables
        required_vars = ["TAVILY_API_KEY", "GEMINI_API_KEY", "GEMINI_BASE_PATH"]
        missing_vars = []
        
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            print("💡 Please add them to your .env file")
            sys.exit(1)
        
        print("✅ All environment variables found")
        
        # Disable tracing
        set_tracing_disabled(True)
        
        # Initialize clients
        print("🔌 Initializing clients...")
        
        key = os.environ.get("GEMINI_API_KEY")
        base_url = os.environ.get("GEMINI_BASE_PATH")
        
        gemini_client = AsyncOpenAI(api_key=key, base_url=base_url)
        gemini_model = OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=gemini_client,
        )
        
        print("✅ Gemini client initialized")
        
        @dataclass
        class AppContext:
            tavily: TavilyClient

        class SearchArgs(BaseModel):
            query: str
            max_results: int | None = 5

        @function_tool
        def tavily_search(ctx: RunContextWrapper[AppContext], args: SearchArgs) -> str:
            """Search the web via Tavily and return JSON with answer and top results."""
            print(f"🔍 Searching for: {args.query}")
            try:
                resp = ctx.context.tavily.search(args.query)
                results = resp.get("results", [])[: (args.max_results or 5)]
                out = {
                    "query": resp.get("query", args.query),
                    "answer": resp.get("answer"),
                    "results": [
                        {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content")}
                        for r in results
                    ],
                }
                return json.dumps(out, ensure_ascii=False)
            except Exception as e:
                print(f"❌ Tavily search error: {e}")
                return json.dumps({"error": str(e)})

        def build_agent() -> Agent[AppContext]:
            return Agent[AppContext](
                name="Web Researcher",
                instructions=(
                    "Use the tavily_search tool for any question that needs fresh or cited facts. "
                    "If tool output already answers the user, summarize it with URLs."
                ),
                model=gemini_model,
                tools=[tavily_search],
            )

        # Main execution
        print("🤖 Building agent...")
        
        api_key = os.environ.get("TAVILY_API_KEY")
        ctx = AppContext(tavily=TavilyClient(api_key=api_key))
        agent = build_agent()
        
        print("✅ Agent built successfully")
        
        # Run the agent
        print("🎯 Running agent query...")
        
        import asyncio
        
        async def run_agent():
            try:
                result = await Runner.run(
                    agent,
                    "Summarize the latest guidance for creating function tools in the OpenAI Agents SDK.",
                    context=ctx,
                )
                print("📋 Result:")
                print(result.final_output)
                return result.final_output
            except Exception as e:
                print(f"❌ Agent execution error: {e}")
                traceback.print_exc()
                return None
        
        # Run the async function
        result = asyncio.run(run_agent())
        
        if result:
            print("🎉 Application completed successfully!")
        else:
            print("❌ Application failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        print("📍 Full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()