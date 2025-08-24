import asyncio
from .graph import graph_app  # relative import from the same package

async def main():
    # Hardcoded query
    state = {
        "original_query": "What is James Webb Telescope?",
        "chat_history": []
    }

    # Run the graph (it has async nodes)
    result = await graph_app.ainvoke(state)

    print("\n=== FINAL ANSWER ===\n")
    if isinstance(result, dict) and "final_answer" in result:
        print(result["final_answer"])
    else:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())