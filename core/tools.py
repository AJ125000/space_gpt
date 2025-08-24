from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools import Tool
import sys
import os
import trafilatura
import json
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
        k=1,  # Set number of results to 3
    )

    def _extract_with_trafilatura(url: str) -> str:
        if trafilatura is None:
            return ""  
        try:
            # Download & extract main content
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return ""
            text = trafilatura.extract(
                downloaded,
                favor_precision=True,
                include_comments=False,
                include_tables=False,
                include_links=False,
            )
            return (text or "").strip()
        except Exception:
            return ""
    
    def _search_and_scrape(query: str) -> str:
        """
        Keeps the same call signature as the original Tool.func.
        Returns a single string: concatenated extracted contents per URL.
        Falls back to the original Serper text if scraping fails.
        """
        # 1) Try to get structured results so we can pull links reliably
        try:
            results = search.results(query)  # structured dict/list
        except Exception:
            # If structured fails, try .run and parse JSON if possible
            try:
                raw = search.run(query)
                results = json.loads(raw)
            except Exception:
                # Last resort: return the raw Serper string
                return raw

        # 2) Collect links from results (handles dict or list shapes)
        links = []
        if isinstance(results, dict):
            organic = results.get("organic", []) or []
            for item in organic:
                link = item.get("link") or item.get("url")
                if link:
                    links.append(link)
        elif isinstance(results, list):
            for item in results:
                link = item.get("link") or item.get("url")
                if link:
                    links.append(link)

        max_items = getattr(search, "k", 3) or 3
        links = links[:max_items]

        # 3) Scrape each link and build the output string
        outputs = []
        max_chars = int(os.getenv("SERPER_SCRAPE_MAX_CHARS", "4000"))
        for idx, url in enumerate(links, 1):
            content = _extract_with_trafilatura(url)
            if not content:
                continue
            if len(content) > max_chars:
                content = content[:max_chars].rstrip() + "…"
            # Keep it textual; include URL as a lightweight header
            outputs.append(f"[{idx}] {url}\n{content}")

        # 4) Fallback: if nothing could be scraped, return original Serper text
        if not outputs:
            try:
                return search.run(query)
            except Exception:
                return ""

        # Concatenate blocks — still a single string return
        return "\n\n---\n\n".join(outputs)
    # Create a tool wrapper for the search functionality
    tool = Tool(
        name="web_search",
        description="Search the web for current information using Serper.dev API. Returns 3 results.",
        func=_search_and_scrape,
    )
    
    return tool

web_search_tool = get_web_search_tool()