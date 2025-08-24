import asyncio
from .graph import graph_app  # relative import from the same package

async def main():
    # Hardcoded query
    state = {
        "original_query": "Who is Abhinav Gothwal?",
        "chat_history": [],
        "rag_query": "",
        "search_query": "",
        "is_out_of_scope": False,
        "retrieved_docs": "",
        "search_results": "",
        "filtered_context": "",
        "final_answer": "",
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