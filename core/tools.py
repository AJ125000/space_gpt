from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools import Tool
import sys
import os

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings

def get_web_search_tool():
    """
    Creates and returns a Serper.dev web search tool.
    Uses the API key from the configuration.
    Returns 3 search results by default.
    """
    print("Initializing Serper Search Tool...")
    
    # Initialize the Serper API wrapper with the API key from config
    search = GoogleSerperAPIWrapper(
        serper_api_key=settings.serper_api_key,
        k=3,  # Set number of results to 3
        snippet_length= 500
    )
    
    # Create a tool wrapper for the search functionality
    tool = Tool(
        name="web_search",
        description="Search the web for current information using Serper.dev API. Returns 3 results.",
        func=search.run,
    )
    
    return tool

web_search_tool = get_web_search_tool()