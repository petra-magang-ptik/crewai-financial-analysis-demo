import json
import os
from typing import Type, ClassVar

from pydantic import BaseModel, Field
import requests
from langchain_community.tools.yahoo_finance_news import (
    YahooFinanceNewsTool as langchain_yfinance_tool,
)
from crewai.tools import BaseTool


class _SearchToolInput(BaseModel):
    """Input schema for search tools."""

    query: str = Field(..., description="The query to search for")


class SearchInternetTool(BaseTool):
    """Search the internet using Serper.dev search endpoint."""

    name: str = "Search internet"
    description: str = (
        "Search the internet about a given topic and return relevant results"
    )
    args_schema: Type[BaseModel] = _SearchToolInput

    def _run(self, query: str) -> str:
        top_result_to_return = 4
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            "X-API-KEY": os.environ.get("SERPER_API_KEY", ""),
            "content-type": "application/json",
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        results = response.json().get("organic", [])
        string = []
        for result in results[:top_result_to_return]:
            try:
                string.append(
                    "\n".join(
                        [
                            f"Title: {result['title']}",
                            f"Link: {result['link']}",
                            f"Snippet: {result['snippet']}",
                            "\n-----------------",
                        ]
                    )
                )
            except KeyError:
                # skip malformed result
                continue

        return "\n".join(string)


class SearchNewsTool(BaseTool):
    """Search news on the internet using Serper.dev news endpoint."""

    name: str = "Search news on the internet"
    description: str = (
        "Search news about a company, stock or any other topic and return relevant results"
    )
    args_schema: Type[BaseModel] = _SearchToolInput

    def _run(self, query: str) -> str:
        top_result_to_return = 4
        url = "https://google.serper.dev/news"
        payload = json.dumps({"q": query})
        headers = {
            "X-API-KEY": os.environ.get("SERPER_API_KEY", ""),
            "content-type": "application/json",
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        results = response.json().get("news", [])
        string = []
        for result in results[:top_result_to_return]:
            try:
                string.append(
                    "\n".join(
                        [
                            f"Title: {result['title']}",
                            f"Link: {result['link']}",
                            f"Snippet: {result['snippet']}",
                            "\n-----------------",
                        ]
                    )
                )
            except KeyError:
                # skip malformed result
                continue

        return "\n".join(string)


class _YahooFinanceNewsToolInput(BaseModel):
    """Input for YahooFinanceNewsTool"""

    query: str = Field(..., description="The query to search for news articles")


class YahooFinanceNewsTool(BaseTool):
    name: str = "Yahoo Finance News"
    description: str = (
        "Search for news articles about a company or stock using Yahoo Finance"
    )
    args_schema: Type[BaseModel] = _YahooFinanceNewsToolInput
    tool: ClassVar = langchain_yfinance_tool()

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.tool.invoke(query)
