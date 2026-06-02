from datetime import UTC, datetime
from typing import Any
from uuid import UUID


def iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.isoformat().replace("+00:00", "Z")


def text_redacted(text: str, text_raw_hash: str | None = None) -> dict[str, Any]:
    return {"text_redacted": text, "text_raw_hash": text_raw_hash}


def session_span(
    *,
    span_id: UUID,
    parent_span_id: UUID | None,
    start_time: datetime,
    end_time: datetime | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "span_kind": "session",
        "span_id": str(span_id),
        "parent_span_id": str(parent_span_id) if parent_span_id else None,
        "start_time": iso_z(start_time),
    }
    if end_time is not None:
        row["end_time"] = iso_z(end_time)
    if outcome is not None:
        row["outcome"] = outcome
    return row


def model_call_span(
    *,
    span_id: UUID,
    parent_span_id: UUID | None,
    model_name: str,
    start_time: datetime,
    end_time: datetime | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "span_kind": "model_call",
        "span_id": str(span_id),
        "parent_span_id": str(parent_span_id) if parent_span_id else None,
        "model_name": model_name,
        "start_time": iso_z(start_time),
    }
    if end_time is not None:
        row["end_time"] = iso_z(end_time)
    if outcome is not None:
        row["outcome"] = outcome
    return row


def tool_call_span(
    *,
    span_id: UUID,
    parent_span_id: UUID | None,
    tool_name: str,
    start_time: datetime,
    end_time: datetime | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "span_kind": "tool_call",
        "span_id": str(span_id),
        "parent_span_id": str(parent_span_id) if parent_span_id else None,
        "tool_name": tool_name,
        "start_time": iso_z(start_time),
    }
    if end_time is not None:
        row["end_time"] = iso_z(end_time)
    if outcome is not None:
        row["outcome"] = outcome
    return row


def session_start_event(
    *, event_id: UUID, span_id: UUID, timestamp: datetime
) -> dict[str, Any]:
    return {
        "event_type": "session_start",
        "event_id": str(event_id),
        "span_id": str(span_id),
        "timestamp": iso_z(timestamp),
    }


def session_end_event(
    *, event_id: UUID, span_id: UUID, timestamp: datetime
) -> dict[str, Any]:
    return {
        "event_type": "session_end",
        "event_id": str(event_id),
        "span_id": str(span_id),
        "timestamp": iso_z(timestamp),
    }


def system_instructions_loaded_event(
    *,
    event_id: UUID,
    span_id: UUID,
    timestamp: datetime,
    instructions: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_type": "system_instructions_loaded",
        "event_id": str(event_id),
        "span_id": str(span_id),
        "timestamp": iso_z(timestamp),
        "instructions": instructions,
    }


def prompt_received_event(
    *,
    event_id: UUID,
    span_id: UUID,
    timestamp: datetime,
    input_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_type": "prompt_received",
        "event_id": str(event_id),
        "span_id": str(span_id),
        "timestamp": iso_z(timestamp),
        "input": input_payload,
    }


def model_call_start_event(
    *,
    event_id: UUID,
    span_id: UUID,
    timestamp: datetime,
    input_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_type": "model_call_start",
        "event_id": str(event_id),
        "span_id": str(span_id),
        "timestamp": iso_z(timestamp),
        "input": input_payload,
    }


def model_call_end_event(
    *,
    event_id: UUID,
    span_id: UUID,
    timestamp: datetime,
    output_payload: dict[str, Any],
    outcome: str,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event_type": "model_call_end",
        "event_id": str(event_id),
        "span_id": str(span_id),
        "timestamp": iso_z(timestamp),
        "output": output_payload,
        "outcome": outcome,
    }
    if error is not None:
        row["error"] = error
    return row


def tool_call_start_event(
    *,
    event_id: UUID,
    span_id: UUID,
    timestamp: datetime,
    args: Any,
) -> dict[str, Any]:
    return {
        "event_type": "tool_call_start",
        "event_id": str(event_id),
        "span_id": str(span_id),
        "timestamp": iso_z(timestamp),
        "args": args,
    }


def tool_call_end_event(
    *,
    event_id: UUID,
    span_id: UUID,
    timestamp: datetime,
    result: Any,
    outcome: str,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event_type": "tool_call_end",
        "event_id": str(event_id),
        "span_id": str(span_id),
        "timestamp": iso_z(timestamp),
        "result": result,
        "outcome": outcome,
    }
    if error is not None:
        row["error"] = error
    return row
