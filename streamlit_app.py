from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple, cast

import html
import re

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from dotenv import load_dotenv

from crewai.events import crewai_event_bus

from listeners import StreamlitCrewListener
from main import FinancialCrew

load_dotenv()

st.set_page_config(
    page_title="Financial Crew Live Monitor",
    page_icon="📈",
    layout="wide",
)


EXECUTOR = ThreadPoolExecutor(max_workers=1)
EXPECTED_TASKS = 4
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
STATUS_STYLES = {
    "pending": ("🕓", "#9ca3af"),
    "running": ("🟡", "#f59e0b"),
    "completed": ("✅", "#16a34a"),
    "failed": ("❌", "#ef4444"),
}


def _normalise_alias(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped.casefold()
    return None


def _merge_agent_output(entry: Dict[str, Any], new_output: Any) -> None:
    if new_output in (None, ""):
        return

    fragments: List[Any] = entry.setdefault("output_fragments", [])

    if fragments:
        last_fragment = fragments[-1]
        if last_fragment == new_output:
            entry["output"] = _serialise_agent_output(fragments)
            return

        if isinstance(last_fragment, str) and isinstance(new_output, str):
            if new_output.strip() and new_output.strip() in last_fragment:
                entry["output"] = _serialise_agent_output(fragments)
                return
            if last_fragment.strip() and last_fragment.strip() in new_output:
                fragments[-1] = new_output
                entry["output"] = _serialise_agent_output(fragments)
                return

    fragments.append(new_output)
    entry["output"] = _serialise_agent_output(fragments)


def _serialise_agent_output(fragments: List[Any]) -> Any:
    if all(isinstance(fragment, str) for fragment in fragments):
        merged = "\n\n".join(fragment for fragment in fragments if fragment)
        return merged
    return fragments[-1]


def _status_visual(status: str) -> tuple[str, str]:
    return STATUS_STYLES.get(status, ("⚪", "#9ca3af"))


def _split_text_with_think_sections(text: str) -> List[Tuple[str, str]]:
    if not text:
        return []

    segments: List[Tuple[str, str]] = []
    last_index = 0

    for match in THINK_PATTERN.finditer(text):
        normal_segment = text[last_index: match.start()]
        if normal_segment.strip():
            segments.append(("markdown", normal_segment))

        think_content = match.group(1).strip()
        if think_content:
            segments.append(("think", think_content))
        last_index = match.end()

    remainder = text[last_index:]
    if remainder.strip():
        segments.append(("markdown", remainder))

    if not segments:
        segments.append(("markdown", text))

    return segments


def _render_text_with_think_sections(
    text: str,
    *,
    target: Optional[DeltaGenerator] = None,
    summary_label: str = "Show hidden reasoning",
    expand_think: bool = False,
) -> None:
    if not text:
        return

    render_target = target or st
    segments = _split_text_with_think_sections(text)

    think_counter = 0
    for kind, content in segments:
        normalised = content.strip("\n")
        if not normalised:
            continue

        if kind == "think":
            think_counter += 1
            label = summary_label if think_counter == 1 else f"{summary_label} ({think_counter})"
            expander = render_target.expander(label, expanded=expand_think)
            with expander:
                expander.markdown(normalised)
        else:
            render_target.markdown(normalised)


def _render_details(
    summary: str,
    text: Any,
    *,
    target: Optional[DeltaGenerator] = None,
    indent_ratio: Optional[float] = None,
    expand: bool = False,
    think_summary_label: str = "Show hidden reasoning",
) -> None:
    render_target = target or st

    if indent_ratio is not None and indent_ratio > 0:
        columns = render_target.columns([indent_ratio, 1])
        details_target = columns[1]
    else:
        details_target = render_target

    expander = details_target.expander(summary, expanded=expand)
    with expander:
        _render_text_with_think_sections(
            str(text),
            target=expander,
            summary_label=think_summary_label,
            expand_think=expand,
        )


def _normalise_tool_args(args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        stripped = args.strip()
        if not stripped:
            return ""
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        else:
            args = parsed
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except TypeError:
        return repr(args)


def _tool_signature(event_data: Dict[str, Any]) -> tuple:
    agent_identifier = next(
        (
            value
            for value in (
                event_data.get("agent_id"),
                event_data.get("agent_key"),
                event_data.get("source_fingerprint"),
            )
            if isinstance(value, str) and value
        ),
        "",
    )

    args_repr = _normalise_tool_args(event_data.get("tool_args"))

    run_attempts = event_data.get("run_attempts")
    if run_attempts in (None, "", 0):
        attempt_repr = "1"
    else:
        attempt_repr = str(run_attempts)

    return (
        agent_identifier,
        attempt_repr,
        args_repr,
    )


def _find_matching_tool_entry(
    tools: Dict[str, Dict[str, Any]], signature: tuple, event_data: Dict[str, Any]
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    for key, entry in tools.items():
        if entry.get("signature") == signature:
            return key, entry

    # Fallback: match by tool name for an active (non-terminal) entry
    name = event_data.get("tool_name") or ""
    for key, entry in tools.items():
        if entry.get("name") == name and entry.get("status") in {"pending", "running"}:
            return key, entry
    agent_identifier, _, args_repr = signature
    for key, entry in tools.items():
        if (
            entry.get("agent_identifier") == agent_identifier
            and entry.get("args_repr") == args_repr
            and entry.get("status") in {"pending", "running"}
        ):
            return key, entry

    return None, None


def _init_session_state() -> None:
    defaults = {
        "run_id": None,
        "status": "idle",
        "events": [],
        "completed_tasks": set(),
        "task_registry": {},
        "agent_registry": {},
        "task_alias_map": {},
        "agent_alias_map": {},
        "event_queue": None,
        "crew_future": None,
        "future_processed": False,
        "final_output": None,
        "errors": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _start_run(company: str) -> None:
    run_id = str(uuid.uuid4())
    event_queue: Queue = Queue()

    st.session_state.update(
        {
            "run_id": run_id,
            "status": "running",
            "events": [],
            "completed_tasks": set(),
            "task_registry": {},
            "agent_registry": {},
            "task_alias_map": {},
            "agent_alias_map": {},
            "event_queue": event_queue,
            "final_output": None,
            "errors": [],
            "future_processed": False,
        }
    )

    future = EXECUTOR.submit(_run_financial_crew, company, run_id, event_queue)
    st.session_state["crew_future"] = future


def _run_financial_crew(company: str, run_id: str, event_queue: Queue) -> None:
    financial_crew = FinancialCrew(company)
    try:
        with crewai_event_bus.scoped_handlers():
            StreamlitCrewListener(run_id=run_id, event_queue=event_queue)
            result = financial_crew.run()

        event_queue.put(
            {
                "run_id": run_id,
                "type": "run:completed",
                "source": "FinancialCrew",
                "event": {"output": result, "timestamp": datetime.utcnow().isoformat()},
            }
        )
    except Exception as exc:  # pragma: no cover - surfaced in UI
        event_queue.put(
            {
                "run_id": run_id,
                "type": "run:failed",
                "source": "FinancialCrew",
                "event": {
                    "error": repr(exc),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }
        )
        raise


def _drain_event_queue() -> None:
    event_queue: Optional[Queue] = st.session_state.get("event_queue")
    if event_queue is None:
        return

    while True:
        try:
            payload = event_queue.get_nowait()
        except Empty:
            break
        else:
            _process_event(payload)


def _process_event(payload: Dict[str, Any]) -> None:
    if payload.get("run_id") != st.session_state.get("run_id"):
        return

    event_type = payload.get("type", "event:unknown")
    event_data = payload.get("event", {})

    st.session_state["events"].append(payload)

    task_id, task_entry = _update_task_registry(event_type, event_data)
    if task_entry is None and task_id:
        task_entry = st.session_state.get("task_registry", {}).get(task_id)

    agent_entry = _update_agent_registry(event_type, event_data, task_id)
    if agent_entry and task_entry is not None:
        agents = task_entry.setdefault("agents", {})
        agents[agent_entry["run_key"]] = agent_entry

    if event_type == "task:completed":
        key = task_id or (task_entry or {}).get("task_id")
        if key:
            st.session_state["completed_tasks"].add(key)
        else:
            st.session_state["completed_tasks"].add(event_type)
    elif event_type == "run:completed":
        st.session_state["status"] = "completed"
        st.session_state["final_output"] = event_data.get("output")
    elif event_type == "run:failed":
        st.session_state["status"] = "failed"
        st.session_state["errors"].append(event_data.get("error", "Unknown error"))


def _update_task_registry(event_type: str, event_data: Dict[str, Any]) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    tasks = st.session_state.setdefault("task_registry", {})
    alias_map = st.session_state.setdefault("task_alias_map", {})

    identifier_aliases = [
        event_data.get("task_id"),
        event_data.get("id"),
    ]
    name_aliases = [
        event_data.get("name"),
        event_data.get("task_name"),
    ]

    alias_candidates = [
        value.strip()
        for value in identifier_aliases + name_aliases
        if isinstance(value, str) and value.strip()
    ]

    task_id: Optional[str] = None

    for alias in alias_candidates:
        alias_key = _normalise_alias(alias)
        if alias_key and alias_key in alias_map:
            task_id = alias_map[alias_key]
            break

    if task_id is None:
        for alias in identifier_aliases:
            if isinstance(alias, str) and alias.strip():
                task_id = alias.strip()
                break

    if task_id is None and name_aliases:
        normalised_names = {
            _normalise_alias(name)
            for name in name_aliases
            if isinstance(name, str) and name.strip()
        }
        for existing_id, entry in tasks.items():
            entry_name = entry.get("name") if isinstance(entry.get("name"), str) else None
            entry_key = _normalise_alias(entry_name) if entry_name else None
            if entry_key and entry_key in normalised_names:
                task_id = existing_id
                break

    if task_id is None and not alias_candidates and not event_type.startswith("task:"):
        return None, None

    if task_id is None and alias_candidates:
        task_id = alias_candidates[0]

    if task_id is None:
        task_id = uuid.uuid4().hex

    for alias in alias_candidates:
        alias_key = _normalise_alias(alias)
        if alias_key:
            alias_map[alias_key] = task_id

    entry = tasks.setdefault(
        task_id,
        {
            "task_id": task_id,
            "name": event_data.get("name") or event_data.get("task_name") or event_data.get("description") or task_id,
            "status": "pending",
            "history": [],
            "agents": {},
            "first_seen": event_data.get("timestamp"),
        },
    )

    entry["task_id"] = task_id

    preferred_name = event_data.get("name") or event_data.get("task_name") or event_data.get("description")
    if preferred_name:
        entry["name"] = preferred_name

    status_lookup = {
        "task:started": "running",
        "task:completed": "completed",
        "task:failed": "failed",
    }
    maybe_status = status_lookup.get(event_type)
    if maybe_status:
        entry["status"] = maybe_status

    entry.setdefault("first_seen", event_data.get("timestamp"))
    entry["last_event"] = event_type
    entry["last_updated"] = event_data.get("timestamp")

    history_entry = {
        "type": event_type,
        "timestamp": event_data.get("timestamp"),
        "summary": {k: v for k, v in event_data.items() if k not in {"timestamp", "output"}},
    }
    entry.setdefault("history", []).append(history_entry)

    return task_id, entry


def _update_agent_registry(
    event_type: str, event_data: Dict[str, Any], task_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    if not event_type.startswith("agent:") and not event_type.startswith("tool:"):
        return None

    task_id = task_id or event_data.get("task_id")
    if not task_id:
        return None

    registry = st.session_state.setdefault("agent_registry", {})
    alias_store = st.session_state.setdefault("agent_alias_map", {})
    task_aliases = alias_store.setdefault(task_id, {})

    alias_candidates = [
        value.strip()
        for value in (
            event_data.get("source_fingerprint"),
            event_data.get("agent_id"),
            event_data.get("agent_key"),
            event_data.get("agent_role"),
        )
        if isinstance(value, str) and value.strip()
    ]

    run_key: Optional[str] = None
    entry: Optional[Dict[str, Any]] = None

    for alias in alias_candidates:
        alias_key = _normalise_alias(alias)
        if not alias_key:
            continue
        candidate_key = task_aliases.get(alias_key)
        if candidate_key and candidate_key in registry:
            run_key = candidate_key
            entry = registry[candidate_key]
            break

    if run_key is None:
        for candidate_key, candidate_entry in registry.items():
            if candidate_entry.get("task_id") != task_id:
                continue
            if event_data.get("agent_id") and candidate_entry.get("agent_id") == event_data.get("agent_id"):
                run_key = candidate_key
                entry = candidate_entry
                break
            if event_data.get("source_fingerprint") and candidate_entry.get("source_fingerprint") == event_data.get("source_fingerprint"):
                run_key = candidate_key
                entry = candidate_entry
                break
            if event_data.get("agent_role") and candidate_entry.get("agent_role") == event_data.get("agent_role"):
                run_key = candidate_key
                entry = candidate_entry
                break

    if run_key is None:
        task_candidates = [candidate_entry for candidate_entry in registry.values() if candidate_entry.get("task_id") == task_id]
        if len(task_candidates) == 1:
            candidate_entry = task_candidates[0]
            candidate_run_key = candidate_entry.get("run_key")
            if candidate_run_key:
                entry = candidate_entry
                run_key = candidate_run_key

    if run_key is None:
        run_key = _build_agent_run_key(task_id, event_data)
        entry = registry.get(run_key)

    if entry is None:
        entry = {
            "run_key": run_key,
            "task_id": task_id,
            "agent_id": event_data.get("agent_id"),
            "agent_role": event_data.get("agent_role") or "Agent",
            "status": "pending",
            "history": [],
            "tools": {},
            "first_seen": event_data.get("timestamp"),
            "source_fingerprint": event_data.get("source_fingerprint"),
        }
        registry[run_key] = entry

    for alias in alias_candidates:
        alias_key = _normalise_alias(alias)
        if alias_key:
            task_aliases[alias_key] = run_key

    entry["run_key"] = run_key
    entry["task_id"] = task_id
    if event_data.get("agent_role"):
        entry["agent_role"] = event_data["agent_role"]
    if event_data.get("agent_id"):
        entry["agent_id"] = event_data["agent_id"]
    if event_data.get("source_fingerprint"):
        entry["source_fingerprint"] = event_data["source_fingerprint"]

    agent_status_lookup = {
        "agent:started": "running",
        "agent:completed": "completed",
        "agent:error": "failed",
    }
    maybe_status = agent_status_lookup.get(event_type)
    if maybe_status:
        entry["status"] = maybe_status

    if event_type == "agent:completed" and event_data.get("output"):
        _merge_agent_output(entry, event_data["output"])
    if event_type == "agent:error" and event_data.get("error"):
        entry["error"] = event_data["error"]

    if event_type.startswith("tool:"):
        _update_tool_usage(entry, event_type, event_data)

    entry.setdefault("first_seen", event_data.get("timestamp"))
    entry["last_event"] = event_type
    entry["last_updated"] = event_data.get("timestamp")
    entry.setdefault("history", []).append(
        {
            "type": event_type,
            "timestamp": event_data.get("timestamp"),
            "summary": {k: v for k, v in event_data.items() if k not in {"timestamp", "output"}},
        }
    )

    return entry


def _build_agent_run_key(task_id: str, event_data: Dict[str, Any]) -> Optional[str]:
    candidates = [
        event_data.get("source_fingerprint"),
        event_data.get("agent_id"),
        event_data.get("agent_key"),
        event_data.get("agent_role"),
    ]

    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return f"{task_id}:{candidate}"

    fallback = event_data.get("timestamp")
    if isinstance(fallback, str) and fallback:
        return f"{task_id}:{fallback}"

    return f"{task_id}:{uuid.uuid4().hex}"


def _update_tool_usage(agent_entry: Dict[str, Any], event_type: str, event_data: Dict[str, Any]) -> None:
    tools = agent_entry.setdefault("tools", {})
    active_map = agent_entry.setdefault("tool_active_map", {})
    agent_entry.setdefault("tool_sequence", 0)

    signature = _tool_signature(event_data)

    tool_key = active_map.get(signature)
    tool_entry: Optional[Dict[str, Any]] = tools.get(tool_key) if tool_key else None

    if tool_entry is None:
        tool_key, tool_entry = _find_matching_tool_entry(tools, signature, event_data)

    if tool_entry is None:
        agent_entry["tool_sequence"] += 1
        tool_key = f"tool-{agent_entry['tool_sequence']}"
        tool_entry = tools.setdefault(
            tool_key,
            {
                "key": tool_key,
                "signature": signature,
                "name": event_data.get("tool_name")
                or (
                    str(event_data.get("tool_class"))
                    if event_data.get("tool_class") and event_type == "tool:started"
                    else "Tool"
                ),
                "status": "pending",
                "started_at": event_data.get("timestamp"),
                "completed_at": None,
                "output": None,
                "error": None,
                "first_seen": event_data.get("timestamp"),
                "history": [],
                "agent_identifier": signature[0],
                "args_repr": signature[2],
                "run_attempt": signature[1],
            },
        )
    else:
        resolved_key = tool_entry.setdefault("key", tool_key or _build_tool_key(event_data))
        tool_key = resolved_key
        tool_entry.setdefault("history", [])
        tool_entry.setdefault("first_seen", event_data.get("timestamp"))
        tool_entry.setdefault("agent_identifier", signature[0])
        tool_entry.setdefault("args_repr", signature[2])
        tool_entry.setdefault("run_attempt", signature[1])

    if tool_entry is None:
        return

    tool_entry = cast(Dict[str, Any], tool_entry)

    # Update mapping for active tool runs when we receive a start event
    if event_type == "tool:started" and tool_key is not None:
        active_map[signature] = tool_key

    tool_entry["signature"] = signature
    tool_entry["agent_identifier"] = signature[0]
    tool_entry["args_repr"] = signature[2]
    tool_entry["run_attempt"] = signature[1]

    if event_data.get("tool_name"):
        tool_entry["name"] = event_data["tool_name"]
    elif event_type == "tool:started" and event_data.get("tool_class"):
        tool_entry["name"] = str(event_data["tool_class"])
    tool_entry.setdefault("name", "Tool")
    if event_type == "tool:started":
        tool_entry["started_at"] = event_data.get("timestamp")
    if event_type in {"tool:finished", "tool:error"} and event_data.get("started_at"):
        tool_entry.setdefault("started_at", event_data.get("started_at"))

    status_lookup = {
        "tool:started": "running",
        "tool:finished": "completed",
        "tool:error": "failed",
    }
    maybe_status = status_lookup.get(event_type)
    if maybe_status:
        tool_entry["status"] = maybe_status

    if event_type == "tool:finished" and event_data.get("output") is not None:
        tool_entry["output"] = event_data["output"]
        tool_entry["completed_at"] = event_data.get("timestamp")
    if event_type == "tool:error" and event_data.get("error") is not None:
        tool_entry["error"] = event_data["error"]
        tool_entry["completed_at"] = event_data.get("timestamp")

    tool_entry["last_event"] = event_type
    tool_entry["last_updated"] = event_data.get("timestamp")
    tool_entry.setdefault("history", []).append(
        {
            "type": event_type,
            "timestamp": event_data.get("timestamp"),
            "summary": {k: v for k, v in event_data.items() if k not in {"timestamp", "output"}},
        }
    )

    if event_type in {"tool:finished", "tool:error"}:
        active_map.pop(signature, None)


def _build_tool_key(event_data: Dict[str, Any]) -> str:
    base = event_data.get("tool_name") or "tool"
    attempt = event_data.get("run_attempts")
    if attempt is None:
        attempt = event_data.get("timestamp") or uuid.uuid4().hex
    return f"{base}:{attempt}"

def _check_future_completion() -> None:
    future: Optional[Future] = st.session_state.get("crew_future")
    if future is None or not future.done() or st.session_state.get("future_processed"):
        return

    try:
        future.result()
    except Exception as exc:  # pragma: no cover - surfaced in UI
        if st.session_state.get("status") != "failed":
            st.session_state["status"] = "failed"
            st.session_state["errors"].append(repr(exc))
    finally:
        st.session_state["future_processed"] = True


def _progress_fraction() -> float:
    if st.session_state.get("status") == "idle":
        return 0.0
    completed = len(st.session_state.get("completed_tasks", set()))
    return min(completed / EXPECTED_TASKS, 0.999 if st.session_state.get("status") == "running" else 1.0)


def _render_event_feed(events: List[Dict[str, Any]]) -> None:
    if not events:
        st.caption("Event feed will appear here once the run starts.")
        return

    label_map = {
        "crew:kickoff-started": "Crew kickoff started",
        "crew:kickoff-completed": "Crew kickoff completed",
        "crew:kickoff-failed": "Crew kickoff failed",
        "task:started": "Task started",
        "task:completed": "Task completed",
        "task:failed": "Task failed",
        "agent:started": "Agent execution started",
        "agent:completed": "Agent execution completed",
        "agent:error": "Agent execution error",
        "tool:started": "Tool started",
        "tool:finished": "Tool finished",
        "tool:error": "Tool error",
        "run:completed": "Run completed",
        "run:failed": "Run failed",
    }

    for event in events:
        event_type = event.get("type", "event:unknown")
        event_data = event.get("event", {})
        timestamp = event_data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp_display = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime("%H:%M:%S")
            except ValueError:
                timestamp_display = timestamp
        else:
            timestamp_display = ""

        badge = label_map.get(event_type, event_type)

        with st.container():
            header = badge if not timestamp_display else f"{badge} · {timestamp_display}"
            st.markdown(f"**{header}**")

            highlight_keys = [
                "crew_name",
                "crew_id",
                "agent_role",
                "task_name",
                "task_id",
                "tool_name",
                "status",
            ]
            highlights = [
                f"{key.replace('_', ' ').title()}: {event_data[key]}"
                for key in highlight_keys
                if event_data.get(key)
            ]
            if highlights:
                st.caption(" · ".join(highlights))

            details = {
                key: value
                for key, value in event_data.items()
                if key not in highlight_keys and key not in {"timestamp"}
            }
            if details:
                with st.expander("Details", expanded=False):
                    st.json(details)


def _render_agent_tree(task_registry: Dict[str, Any]) -> None:
    if not task_registry:
        st.caption("Agent activity will appear here once the run starts.")
        return

    tasks_sorted = sorted(
        task_registry.values(),
        key=lambda item: (item.get("first_seen") or "", item.get("task_id", "")),
    )

    for task in tasks_sorted:
        task_status = task.get("status", "pending")
        task_icon, task_color = _status_visual(task_status)
        task_name = task.get("name") or task.get("task_id") or "Task"
        task_header = (
            f"<div style='margin-bottom:0.5rem;'>"
            f"<div><strong>{task_icon} Task: {html.escape(task_name)}</strong></div>"
            f"<div style='margin-left:1.2rem;color:{task_color};font-size:0.85rem;'>Status: {task_status.title()}</div>"
            f"</div>"
        )
        task_container = st.container()
        task_container.markdown(task_header, unsafe_allow_html=True)

        agents = sorted(
            task.get("agents", {}).values(),
            key=lambda item: (item.get("first_seen") or "", item.get("agent_role") or ""),
        )
        if not agents:
            task_container.markdown(
                "<div style='margin-left:1.2rem;color:#9ca3af;font-size:0.85rem;'>Awaiting agent activity…</div>",
                unsafe_allow_html=True,
            )
            continue

        for agent in agents:
            agent_status = agent.get("status", "pending")
            agent_icon, agent_color = _status_visual(agent_status)
            agent_role = agent.get("agent_role") or "Agent"
            agent_header = (
                f"<div style='margin-left:1.2rem;margin-bottom:0.25rem;'>"
                f"<div><strong>{agent_icon} Agent: {html.escape(agent_role)}</strong></div>"
                f"<div style='margin-left:1.2rem;color:{agent_color};font-size:0.8rem;'>Status: {agent_status.title()}</div>"
                f"</div>"
            )
            agent_container = task_container.container()
            agent_container.markdown(agent_header, unsafe_allow_html=True)

            if agent.get("output"):
                _render_details(
                    "Agent output",
                    agent["output"],
                    target=agent_container,
                    indent_ratio=0.04,
                )

            if agent.get("error"):
                agent_container.markdown(
                    f"<div style='margin-left:2.4rem;color:#ef4444;font-size:0.85rem;'>Error: {html.escape(str(agent['error']))}</div>",
                    unsafe_allow_html=True,
                )

            tools = sorted(
                agent.get("tools", {}).values(),
                key=lambda item: (item.get("first_seen") or "", item.get("name") or ""),
            )
            for tool in tools:
                tool_status = tool.get("status", "pending")
                tool_icon, tool_color = _status_visual(tool_status)
                tool_name = tool.get("name") or "Tool"
                tool_header = (
                    f"<div style='margin-left:2.4rem;margin-bottom:0.15rem;'>"
                    f"<div>{tool_icon} {html.escape(tool_name)}"
                    f"<span style='color:{tool_color};font-size:0.75rem;margin-left:0.5rem;'>{tool_status.title()}</span></div>"
                    f"</div>"
                )
                tool_container = agent_container.container()
                tool_container.markdown(tool_header, unsafe_allow_html=True)

                if tool.get("output"):
                    _render_details(
                        "Tool output",
                        str(tool["output"]),
                        target=tool_container,
                        indent_ratio=0.04,
                    )
                if tool.get("error") is not None:
                    tool_container.markdown(
                        "<div style='margin-left:3.2rem;color:#ef4444;font-size:0.75rem;margin-bottom:0.2rem;'>Error encountered</div>",
                        unsafe_allow_html=True,
                    )
                    _render_details(
                        "Error details",
                        tool["error"],
                        target=tool_container,
                        indent_ratio=0.04,
                    )


def _render_status_panel(company: str) -> None:
    status = st.session_state.get("status", "idle")
    progress = _progress_fraction()
    status_badge = {
        "idle": "grey",
        "running": "orange",
        "completed": "green",
        "failed": "red",
    }.get(status, "grey")

    st.markdown(f"### Run status")
    st.markdown(f"<span style='color:{status_badge};font-size:1.1rem;'>●</span> **{status.title()}**", unsafe_allow_html=True)
    st.progress(progress)
    st.caption(f"{len(st.session_state.get('completed_tasks', []))}/{EXPECTED_TASKS} tasks completed")
    st.caption(f"Tracking company: **{company}**")


def _render_output_panel() -> None:
    if st.session_state.get("status") == "failed" and st.session_state.get("errors"):
        st.error("\n\n".join(st.session_state["errors"]))

    if st.session_state.get("final_output"):
        st.markdown("### Final recommendation")
        _render_text_with_think_sections(str(st.session_state["final_output"]))


def main() -> None:
    _init_session_state()

    st.title("📈 Financial Crew Live Monitor")
    st.caption("Visualise CrewAI agents, tasks, and tools in real time while the analysis runs.")

    model = os.getenv("MODEL", "<unset>")
    base_url = os.getenv("MODEL_BASE_URL", "<unset>")

    with st.sidebar:
        st.subheader("Configuration")
        company_default = st.session_state.get("last_company", "AAPL")
        company_input = st.text_input("Company or ticker", value=company_default)
        company = company_input or ""
        st.session_state["last_company"] = company

        st.markdown("---")
        st.caption("LLM settings")
        st.code(f"MODEL={model}\nMODEL_BASE_URL={base_url}")

        run_disabled = st.session_state.get("status") == "running"
        if st.button("Run analysis", type="primary", disabled=run_disabled):
            cleaned_company = company.strip()
            if cleaned_company:
                _start_run(cleaned_company)
                st.rerun()
            else:
                st.warning("Please provide a company or ticker symbol.")

    _drain_event_queue()
    _check_future_completion()

    left, right = st.columns([0.5, 0.5])

    with left:
        _render_status_panel(company or "(not set)")
        _render_output_panel()

    with right:
        tab_tree, tab_feed = st.tabs(["Agent timeline", "Event feed"])

        with tab_tree:
            st.markdown("### Agent timeline")
            _render_agent_tree(st.session_state.get("task_registry", {}))

        with tab_feed:
            st.markdown("### Event feed")
            _render_event_feed(st.session_state.get("events", []))

    if st.session_state.get("status") == "running":
        time.sleep(0.2)
        st.rerun()


if __name__ == "__main__":
    main()