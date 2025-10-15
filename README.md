
 # Financial Analysis Crew

Welcome! This project shows how to orchestrate a small team of AI agents to research a public company, crunch the numbers, read SEC filings, and produce an investment-style brief. It uses [CrewAI](https://github.com/joaomdmoura/crewai) with tools for web search, financial news, arithmetic, and EDGAR filings, plus an optional Streamlit dashboard so you can watch the agents work in real time.

> Baca versi Bahasa Indonesia: [README.id.md](./README.id.md)

If you're new to Python—no worries. The steps below walk you through installing everything, running the command-line experience, and launching the dashboard UI.

---

## What you get

- **CLI assistant** that asks for a ticker or company name, then generates a research report by coordinating several specialist agents.
- **Streamlit dashboard** (`streamlit_app.py`) that lets you monitor agent progress, see tool usage, and read the final recommendation from your browser.
- **Reusable tools** for web/news search, calculator-style math, and SEC filing retrieval that you can plug into other CrewAI projects.

---

## Prerequisites

- **Python 3.10 or later.** Install from [python.org](https://www.python.org/downloads/) if you do not already have it.
- **A package manager.** This project was tested with [`uv`](https://docs.astral.sh/uv/), but plain `pip` works too. Beginners may prefer `uv` because it handles virtual environments automatically.
- **An Ollama-compatible model** (for example, [Ollama](https://ollama.com/)) running somewhere reachable, or another LangChain-compatible endpoint that exposes the same API.
- **API keys:**
	- `SERPER_API_KEY` to enable Google-like search and news lookup via [Serper.dev](https://serper.dev/).
	- `EDGAR_IDENTITY` (or `SEC_IDENTITY`/`SEC_CONTACT`) — required by the SEC for accessing EDGAR filings. This should be an email address you control.


---

## Quick start (recommended: uv)

```bash
# Install uv if you need it (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the project root
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, creates an isolated environment, and installs every dependency. When the command finishes you can run the app with `uv run ...` commands (examples below).

### Prefer pip?

```bash
# Create and activate a virtual environment (Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install crewai langchain-ollama edgartools streamlit python-dotenv requests
```

> **Tip:** If you're on Windows, replace `source .venv/bin/activate` with `.venv\Scripts\activate`.

---

## Configure environment variables

Create a `.env` file in the project root to keep secrets out of your shell history:

```bash
cp .env.example .env  # if you create one
```

Then fill in the values. You can paste this template into `.env`:

```ini
# Required for connecting to your LLM
MODEL=llama3.1:8b
MODEL_BASE_URL=http://localhost:11434

# Optional but recommended tools
SERPER_API_KEY=your_serper_key
EDGAR_IDENTITY=you@example.com
EMBEDDING_MODEL=llama3.1:8b
SEC_IDENTITY=you@example.com       # alias accepted by the SEC tool

# Streamlit tweaks (optional)
STREAMLIT_SERVER_PORT=8501
```

When you run `main.py` or `streamlit_app.py`, the project loads `.env` automatically via `python-dotenv`.

---

## Run the command-line experience

```bash
uv run python main.py
```

You will be prompted for a company or ticker symbol (for example, `AAPL`). The script spins up a `FinancialCrew` with:

1. A **research analyst** agent (news + sentiment).
2. A **financial analyst** agent (fundamentals and peer comparison).
3. An **investment advisor** agent (recommendations).

They collaborate through CrewAI and produce a formatted report printed to your terminal.

---

## Watch the agents in Streamlit

```bash
uv run streamlit run streamlit_app.py
```

Open the printed URL (usually `http://localhost:8501`) and type a ticker. The dashboard shows:

- Agent/task timelines updated in real-time.
- Tool calls with inputs/outputs.
- The final recommendation, with hidden “think” sections you can reveal.

Stop the app with `Ctrl+C` inside the terminal when you are done.

---

## Project layout

```
├── main.py               # CLI entry point for running the crew
├── streamlit_app.py      # Streamlit UI to observe runs live
├── agents.py             # Agent definitions and tool wiring
├── tasks.py              # CrewAI tasks for research, analysis, filings, recommendation
├── listeners.py          # Streamlit event bridge for real-time updates
├── tools/                # Calculator, web/news search, SEC filings helper tools
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Locked dependency versions (used by uv)
└── README.md
```

---

## How it works (under the hood)

1. `main.py` creates the `FinancialCrew`, wires up agents with their tools, and kicks off the run.
2. `agents.py` defines each agent’s role, goal, and tools. Tools include:
	 - Web/news search via Serper.dev.
	 - Yahoo! Finance headlines via LangChain.
	 - Calculator for quick arithmetic.
	 - EDGAR filing search that chunks documents and uses embeddings for retrieval.
3. `tasks.py` describes what each agent should deliver and how their outputs build toward the final recommendation.
4. `streamlit_app.py` subscribes to CrewAI events, streams them into a queue, and renders them with collapsible sections. You can see intermediate reasoning and tool usage as it happens.

---

## Troubleshooting

- **`MODEL` or `MODEL_BASE_URL` missing:** Double-check your `.env`. If you do not have an Ollama server, point `MODEL_BASE_URL` to the service that hosts your chosen model.
- **`EDGAR identity` error:** The SEC requires a contact email for automated requests. Set `EDGAR_IDENTITY=you@example.com`.
- **Serper-related errors:** Either remove the search tools from `agents.py` or supply a valid `SERPER_API_KEY` from your Serper.dev dashboard.
- **Slow or no responses:** Large filings can take time to download and embed. Try a different ticker or ensure your embedding model is running locally.

---

## Extend the project

- Swap in different CrewAI agents or tools (e.g., add sentiment analysis or charting).
- Log agent outputs to a database for later review.
- Deploy the Streamlit app to [Streamlit Community Cloud](https://streamlit.io/cloud) once your API keys are stored securely.

If you are just getting started with Python, keep experimenting! The more you run and tweak the code, the faster everything will feel familiar.

Happy analyzing! 📈
