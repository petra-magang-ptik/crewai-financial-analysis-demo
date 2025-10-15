from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from queue import Queue

EXCLUDED_FIELDS = {
    "task",
    "from_task",
    "from_agent",
    "agent",
    "tool",
    "tool_instance",
    "crew",
    "crew_instance",
    "llm",
    "callbacks",
    "memory",
    "knowledge",
}

from crewai.events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
    BaseEventListener,
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.utilities.serialization import to_serializable


class StreamlitCrewListener(BaseEventListener):
    """Feeds CrewAI lifecycle events into a thread-safe queue for visualization."""

    def __init__(self, run_id: str, event_queue: Queue):
        self._run_id = run_id
        self._queue = event_queue
        super().__init__()

    def setup_listeners(self, crewai_event_bus) -> None:
        crewai_event_bus.on(CrewKickoffStartedEvent)(self._build_handler("crew:kickoff-started"))
        crewai_event_bus.on(CrewKickoffCompletedEvent)(self._build_handler("crew:kickoff-completed"))
        crewai_event_bus.on(CrewKickoffFailedEvent)(self._build_handler("crew:kickoff-failed"))

        crewai_event_bus.on(TaskStartedEvent)(self._build_handler("task:started"))
        crewai_event_bus.on(TaskCompletedEvent)(self._build_handler("task:completed"))
        crewai_event_bus.on(TaskFailedEvent)(self._build_handler("task:failed"))

        crewai_event_bus.on(AgentExecutionStartedEvent)(self._build_handler("agent:started"))
        crewai_event_bus.on(AgentExecutionCompletedEvent)(self._build_handler("agent:completed"))
        crewai_event_bus.on(AgentExecutionErrorEvent)(self._build_handler("agent:error"))

        crewai_event_bus.on(ToolUsageStartedEvent)(self._build_handler("tool:started"))
        crewai_event_bus.on(ToolUsageFinishedEvent)(self._build_handler("tool:finished"))
        crewai_event_bus.on(ToolUsageErrorEvent)(self._build_handler("tool:error"))

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _build_handler(self, event_type: str):
        def handler(source: Any, event: Any) -> None:
            try:
                payload = {
                    "run_id": self._run_id,
                    "type": event_type,
                    "source": self._describe_source(source),
                    "event": self._enrich_event(event, self._serialise_event(event)),
                }
            except Exception as exc:  # pragma: no cover - defensive guard
                payload = {
                    "run_id": self._run_id,
                    "type": "listener:error",
                    "source": self._describe_source(source),
                    "event": {
                        "error": repr(exc),
                        "event_type": type(event).__name__,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                }
            self._queue.put(payload)

        return handler

    def _enrich_event(self, event: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """Attach lightweight identifiers that survive exclusion filters."""

        enriched = dict(data)

        task = getattr(event, "task", None)
        if task:
            task_id = self._get_task_identifier(task)
            task_name = self._get_task_name(task)
            if task_id:
                enriched.setdefault("task_id", task_id)
            if task_name:
                enriched.setdefault("task_name", task_name)

        agent = getattr(event, "agent", None)
        if agent:
            agent_id, agent_role = self._get_agent_identifiers(agent)
            if agent_id:
                enriched.setdefault("agent_id", agent_id)
            if agent_role:
                enriched.setdefault("agent_role", agent_role)

        return enriched

    def _get_task_identifier(self, task: Any) -> Optional[str]:
        for attr in ("id", "task_id", "name"):
            value = getattr(task, attr, None)
            if isinstance(value, str) and value:
                return value

        fingerprint = getattr(task, "fingerprint", None)
        uuid = getattr(fingerprint, "uuid_str", None)
        if isinstance(uuid, str) and uuid:
            return uuid

        return None

    def _get_task_name(self, task: Any) -> Optional[str]:
        for attr in ("name", "title", "description"):
            value = getattr(task, attr, None)
            if isinstance(value, str) and value:
                return value
        return None

    def _get_agent_identifiers(self, agent: Any) -> tuple[Optional[str], Optional[str]]:
        agent_id = None
        fingerprint = getattr(agent, "fingerprint", None)
        agent_id = getattr(fingerprint, "uuid_str", None)
        if not isinstance(agent_id, str) or not agent_id:
            agent_id = getattr(agent, "id", None)
            if not isinstance(agent_id, str) or not agent_id:
                agent_id = None

        agent_role = getattr(agent, "role", None)
        if not isinstance(agent_role, str) or not agent_role:
            agent_role = getattr(agent, "name", None)
            if not isinstance(agent_role, str) or not agent_role:
                agent_role = None

        return agent_id, agent_role

    def _serialise_event(self, event: Any) -> Dict[str, Any]:
        exclude = EXCLUDED_FIELDS
        if hasattr(event, "to_json"):
            try:
                data = event.to_json(exclude=exclude)
            except Exception:
                data = to_serializable(event, exclude=exclude, max_depth=3)
        else:
            data = to_serializable(event, exclude=exclude, max_depth=3)

        if isinstance(data, dict):
            data.setdefault("event_type", type(event).__name__)
            return data

        return {
            "event_type": type(event).__name__,
            "value": data,
        }

    def _describe_source(self, source: Any) -> str:
        try:
            if hasattr(source, "role") and source.role:
                return f"{type(source).__name__}({source.role})"
            if hasattr(source, "name") and source.name:
                return f"{type(source).__name__}({source.name})"
        except Exception:
            pass
        return type(source).__name__
