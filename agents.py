import imp
from crewai import LLM, Agent
from langchain_ollama import OllamaLLM

import os

from tools.calculator import CalculatorTool
from tools.search import *
from tools.sec import *

tool_search_internet = SearchInternetTool()
tool_search_news = SearchNewsTool()
tool_search_yahoo_finance = YahooFinanceNewsTool()
tool_calculator = CalculatorTool()
tool_search_10q = Search10QTool()
tool_search_10k = Search10KTool()


class StockAnalysisAgents:
    def __init__(self):
        self.llm = OllamaLLM(
            model=os.environ["MODEL"],
            base_url=os.environ["MODEL_BASE_URL"],
        )

    def financial_analyst(self):
        return Agent(
            role="The Best Financial Analyst",
            goal="""Impress all customers with your financial data 
      and market trends analysis""",
            backstory="""The most seasoned financial analyst with 
      lots of expertise in stock market analysis and investment
      strategies that is working for a super important customer. Utilize the given tools to research
      and analyze financial data to provide comprehensive insights""",
            verbose=True,
            tools=[
                tool_search_internet,
                tool_calculator,
                tool_search_10q,
                tool_search_10k,
            ],
            llm=self.llm,
        )

    def research_analyst(self):
        return Agent(
            role="Staff Research Analyst",
            goal="""Being the best at gather, interpret data and amaze
      your customer with it. Utilize the given tools to research
      and analyze financial data to provide comprehensive insights.""",
            backstory="""Known as the BEST research analyst, you're
      skilled in sifting through news, company announcements, 
      and market sentiments. Now you're working on a super 
      important customer.""",
            verbose=True,
            tools=[
                tool_search_internet,
                tool_search_news,
                tool_search_yahoo_finance,
                tool_search_10q,
                tool_search_10k,
            ],
            llm=self.llm,
        )

    def investment_advisor(self):
        return Agent(
            role="Private Investment Advisor",
            goal="""Impress your customers with full analyses over stocks
      and completer investment recommendations. Utilize the given tools to research
      and analyze financial data to provide comprehensive insights.""",
            backstory="""You're the most experienced investment advisor
      and you combine various analytical insights to formulate
      strategic investment advice. You are now working for
      a super important customer you need to impress.""",
            verbose=True,
            tools=[
                tool_search_internet,
                tool_search_news,
                tool_calculator,
                tool_search_yahoo_finance,
            ],
            llm=self.llm,
        )
