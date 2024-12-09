import json
import os
import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
GOOGLE_CX = os.environ["GOOGLE_CX"]


class ToolError(Exception):
    """Raised when a tool encounters an error."""


async def google_search(query: str, count: int = 5) -> str:
    """
    Perform a web search using the Google Custom Search JSON API asynchronously.

    Args:
        query (str): The search query.
        count (int): Number of search results to return.

    Returns:
        str: A JSON list containing search result names, URLs, and snippets.
    """
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "num": count,
    }

    endpoint = "https://www.googleapis.com/customsearch/v1"

    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, params=params, timeout=10) as response:
            if response.status != 200:
                raise ToolError(f"Error: {response.status}, {await response.text()}")

            data = await response.json()
            search_results = data.get("items", [])
            return json.dumps(
                [
                    {
                        "name": item["title"],
                        "url": item["link"],
                        "snippet": item.get("snippet", "No snippet available."),
                    }
                    for item in search_results
                ],
                indent=2,
            )


async def website_text_scraper(url: str) -> str:
    """
    Scrape the text content of a website asynchronously.

    Args:
        url (str): The URL of the website to scrape.

    Returns:
        str: The text content of the website.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                html_content = await response.text()

                soup = BeautifulSoup(html_content, "html.parser")
                paragraphs = soup.find_all("p")
                full_text = " ".join([p.get_text() for p in paragraphs])
                return full_text  # Return the full scraped content
    except Exception as e:
        raise ToolError(f"Error scraping website: {e}") from e
