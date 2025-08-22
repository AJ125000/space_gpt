from langchain_community.tools import DuckDuckGoSearchRun

def get_web_search_tool():
    """
    Creates and returns a DuckDuckGo web search tool.
    This tool does not require an API key.
    """
    print("Initializing DuckDuckGo Search Tool...")
    return DuckDuckGoSearchRun()

web_search_tool = get_web_search_tool()