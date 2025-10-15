import os
from typing import Type, cast

from pydantic import BaseModel, Field

from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from edgar import Company, set_identity
from edgar.entity.filings import EntityFiling

from crewai.tools import BaseTool


class _SECToolInput(BaseModel):
    """Common input schema: "TICKER|QUESTION"""

    query: str = Field(..., description="Pipe-separated: TICKER|QUESTION")


_IDENTITY_CACHE: dict[str, str] = {}


def _ensure_identity() -> str | None:
    identity = (
        os.environ.get("EDGAR_IDENTITY")
        or os.environ.get("SEC_IDENTITY")
        or os.environ.get("SEC_CONTACT")
    )
    if not identity:
        return (
            "EDGAR identity email missing. Set 'EDGAR_IDENTITY' (or SEC_IDENTITY/SEC_CONTACT) "
            "environment variable so requests comply with SEC requirements."
        )

    cached = _IDENTITY_CACHE.get("value")
    if cached == identity:
        return None

    try:
        set_identity(identity)
    except Exception as exc:  # pragma: no cover - defensive
        return f"Failed to configure EDGAR identity: {exc}"

    _IDENTITY_CACHE["value"] = identity
    return None


def _embedding_search(content: str, ask: str) -> str:
    if not content:
        return "Couldn't retrieve filing content for analysis."
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([content])
    if not docs:
        return "Filing content couldn't be segmented for retrieval."
    retriever = FAISS.from_documents(
        docs,
        OllamaEmbeddings(
            model=os.environ["EMBEDDING_MODEL"], base_url=os.environ["MODEL_BASE_URL"]
        ),
    ).as_retriever()
    answers = retriever.get_relevant_documents(ask, top_k=4)
    answers = "\n\n".join([a.page_content for a in answers])
    return answers or "No relevant sections found in the filing."


def _search_latest_form(stock: str, form: str, ask: str) -> str:
    identity_error = _ensure_identity()
    if identity_error:
        return identity_error

    ticker = stock.strip()
    if not ticker:
        return "Ticker symbol is missing. Provide the ticker before the question."

    try:
        company = Company(ticker)
    except Exception as exc:  # pragma: no cover - depends on remote state
        return f"Error locating company for ticker '{ticker}': {exc}"

    if getattr(company, "not_found", False):
        return f"Sorry, I couldn't find any company information for ticker '{ticker}'."

    try:
        filings = company.get_filings(form=form, amendments=False)
    except Exception as exc:  # pragma: no cover - remote call
        return f"Failed to retrieve {form} filings for '{ticker}': {exc}"

    if len(filings) == 0:
        return (
            f"No {form} filings found for ticker '{ticker}'. The company may not have filed "
            f"a {form} yet or it might use a different form type."
        )

    try:
        filing = cast(EntityFiling, filings.latest())
    except Exception as exc:  # pragma: no cover - library edge case
        return f"Unable to determine the latest {form} filing for '{ticker}': {exc}"

    try:
        content = filing.text()
    except Exception as text_exc:  # pragma: no cover - network edge cases
        try:
            content = filing.html() or ""
        except Exception:
            content = ""
        if not content:
            return f"Couldn't download the {form} filing content: {text_exc}"

    context = _embedding_search(content, ask)
    header = (
        f"Ticker: {ticker.upper()}\n"
        f"Company: {getattr(filing, 'company', 'Unknown')}\n"
        f"Form: {filing.form} | Filed: {filing.filing_date}\n"
        f"URL: {filing.filing_url}\n\n"
    )
    return header + context


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
        return _search_latest_form(stock, "10-Q", ask)


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
        return _search_latest_form(stock, "10-K", ask)
