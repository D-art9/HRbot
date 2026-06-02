from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid7

from .client import CyviaClient


def new_id() -> UUID:
    return uuid7()


@dataclass
class Trace:
    client: CyviaClient
    adapter: str
    runtime: str
    agent_name: str
    adapter_version: str | None = None
    runtime_version: str | None = None
    external_agent_id: str | None = None
    agent_description: str | None = None
    agent_display_name: str | None = None
    trace_id: UUID = field(default_factory=new_id)
    root_span_id: UUID = field(default_factory=new_id)
    model_name: str = "unknown"

    _span_buffer: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _event_buffer: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _started: bool = field(default=False, repr=False)
    _session_opened: bool = field(default=False, repr=False)
    _closed: bool = field(default=False, repr=False)

    def merge_system_message(self, data: dict[str, Any]) -> None:
        if self._started:
            return
        if (m := data.get("model")) is not None:
            self.model_name = str(m)
        if (v := data.get("claude_code_version")) is not None:
            self.adapter_version = str(v)
        if (sid := data.get("session_id")) is not None:
            self.external_agent_id = str(sid)

    def start(self) -> None:
        if self._started:
            return
        body = {
            "trace_id": str(self.trace_id),
            "root_span_id": str(self.root_span_id),
            "agent_name": self.agent_name,
            "external_agent_id": self.external_agent_id,
            "agent_description": self.agent_description,
            "agent_display_name": self.agent_display_name,
            "runtime": self.runtime,
            "runtime_version": self.runtime_version,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
        }
        response = self.client.post_json("/agent/traces", body)
        response.raise_for_status()
        self._started = True

    def ensure_started(self) -> None:
        self.start()

    def begin_session(self) -> None:
        if self._session_opened:
            return
        from . import wire

        self.ensure_started()
        now = datetime.now(UTC)
        self.add_span(
            wire.session_span(
                span_id=self.root_span_id,
                parent_span_id=None,
                start_time=now,
            )
        )
        self.add_event(
            wire.session_start_event(
                event_id=new_id(),
                span_id=self.root_span_id,
                timestamp=now,
            )
        )
        self.flush()
        self._session_opened = True

    def add_span(self, span: dict[str, Any]) -> None:
        self._span_buffer.append(span)

    def add_event(self, event: dict[str, Any]) -> None:
        self._event_buffer.append(event)

    def flush(self) -> None:
        if not self._span_buffer and not self._event_buffer:
            return
        spans = self._span_buffer
        events = self._event_buffer
        self._span_buffer = []
        self._event_buffer = []
        body = {"schema_version": 1, "spans": spans, "events": events}
        response = self.client.post_json(f"/agent/traces/{self.trace_id}/ingest", body)
        response.raise_for_status()

    def close(self, *, now: datetime | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._session_opened:
            return
        from . import wire

        ts = now or datetime.now(UTC)
        self.add_event(
            wire.session_end_event(
                event_id=new_id(),
                span_id=self.root_span_id,
                timestamp=ts,
            )
        )
        self.flush()
