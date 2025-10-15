import os

import requests
from typing import Type

from pydantic import BaseModel, Field

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from sec_api import QueryApi
from unstructured.partition.html import partition_html

from crewai.tools import BaseTool


class _SECToolInput(BaseModel):
    """Common input schema: "TICKER|QUESTION"""

    query: str = Field(..., description="Pipe-separated: TICKER|QUESTION")


def _download_form_html(url: str) -> str:
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
        "Cache-Control": "max-age=0",
        "Dnt": "1",
        "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    response = requests.get(url, headers=headers)
    return response.text


def _embedding_search(url: str, ask: str) -> str:
    text = _download_form_html(url)
    elements = partition_html(text=text)
    content = "\n".join([str(el) for el in elements])
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([content])
    retriever = FAISS.from_documents(docs, OllamaEmbeddings(model="mxbai-embed-large")).as_retriever()
    answers = retriever.get_relevant_documents(ask, top_k=4)
    answers = "\n\n".join([a.page_content for a in answers])
    return answers


class Search10QTool(BaseTool):
    """Search the latest 10-Q filing for a ticker and answer a question."""

    name: str = "Search 10-Q form"
    description: str = (
        "Search information from the latest 10-Q form for a given stock. "
        "Input should be 'TICKER|QUESTION' (e.g. 'AAPL|what was last quarter's revenue')"
    )
    args_schema: Type[BaseModel] = _SECToolInput

    def _run(self, query: str) -> str:
        try:
            stock, ask = query.split("|", 1)
        except ValueError:
            return "Input must be 'TICKER|QUESTION'"

        queryApi = QueryApi(api_key=os.environ.get("SEC_API_API_KEY", ""))
        q = {
            "query": {"query_string": {"query": f'ticker:{stock} AND formType:"10-Q"'}},
            "from": "0",
            "size": "1",
            "sort": [{"filedAt": {"order": "desc"}}],
        }
        resp = queryApi.get_filings(q)
        fillings = resp.get("filings", []) if isinstance(resp, dict) else []
        if len(fillings) == 0:
            return "Sorry, I couldn't find any filing for this stock. Check if the ticker is correct."
        link = fillings[0].get("linkToFilingDetails")
        if not link:
            return "Couldn't find a link to the filing details."

        return _embedding_search(link, ask)


class Search10KTool(BaseTool):
    """Search the latest 10-K filing for a ticker and answer a question."""

    name: str = "Search 10-K form"
    description: str = (
        "Search information from the latest 10-K form for a given stock. "
        "Input should be 'TICKER|QUESTION' (e.g. 'AAPL|what was last year's revenue')"
    )
    args_schema: Type[BaseModel] = _SECToolInput

    def _run(self, query: str) -> str:
        try:
            stock, ask = query.split("|", 1)
        except ValueError:
            return "Input must be 'TICKER|QUESTION'"

        queryApi = QueryApi(api_key=os.environ.get("SEC_API_API_KEY", ""))
        q = {
            "query": {"query_string": {"query": f'ticker:{stock} AND formType:"10-K"'}},
            "from": "0",
            "size": "1",
            "sort": [{"filedAt": {"order": "desc"}}],
        }
        resp = queryApi.get_filings(q)
        fillings = resp.get("filings", []) if isinstance(resp, dict) else []
        if len(fillings) == 0:
            return "Sorry, I couldn't find any filing for this stock. Check if the ticker is correct."
        link = fillings[0].get("linkToFilingDetails")
        if not link:
            return "Couldn't find a link to the filing details."

        return _embedding_search(link, ask)
