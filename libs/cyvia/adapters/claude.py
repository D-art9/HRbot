from __future__ import annotations

import types
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .. import wire
from ..trace import Trace, new_id


@dataclass
class ClaudeRecorderState:
    pending_tools: dict[str, dict[str, Any]] = field(default_factory=dict)


def observe(client: Any, trace: Trace) -> None:
    if hasattr(client, "_cyvia"):
        if getattr(client._cyvia, "observed", False):
            return

    client._cyvia = types.SimpleNamespace()
    client._cyvia.trace = trace
    client._cyvia.state = ClaudeRecorderState()
    client._cyvia.orig_query = client.query
    client._cyvia.orig_receive_response = client.receive_response

    client.query = types.MethodType(__wrapped_query, client)
    client.receive_response = types.MethodType(__wrapped_receive_response, client)

    client._cyvia.observed = True


async def __wrapped_query(self: Any, *args: Any, **kwargs: Any) -> Any:
    result = await self._cyvia.orig_query(*args, **kwargs)

    query = args[0] if args else kwargs.get("query")
    record_query(query, self._cyvia.trace)

    return result


async def __wrapped_receive_response(self: Any, *args: Any, **kwargs: Any) -> Any:
    stream = self._cyvia.orig_receive_response(*args, **kwargs)
    try:
        async for message in stream:
            record_message(message, self._cyvia.trace, self._cyvia.state)
            yield message
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()

        self._cyvia.trace.close()


def record_query(query: Any, trace: Trace) -> None:
    if isinstance(query, str):
        trace.begin_session()
        trace.add_event(
            wire.prompt_received_event(
                event_id=new_id(),
                span_id=trace.root_span_id,
                timestamp=datetime.now(UTC),
                input_payload=wire.text_redacted(query),
            )
        )
        trace.flush()


def record_user_message(message: Any, trace: Trace, state: ClaudeRecorderState) -> None:
    trace.begin_session()
    content = message.content
    if isinstance(content, str):
        trace.add_event(
            wire.prompt_received_event(
                event_id=new_id(),
                span_id=trace.root_span_id,
                timestamp=datetime.now(UTC),
                input_payload=wire.text_redacted(content),
            )
        )
        trace.flush()
        return

    if isinstance(content, list):
        for block in content:
            block_name = type(block).__name__
            match block_name:
                case "TextBlock":
                    trace.add_event(
                        wire.prompt_received_event(
                            event_id=new_id(),
                            span_id=trace.root_span_id,
                            timestamp=datetime.now(UTC),
                            input_payload=wire.text_redacted(block.text),
                        )
                    )
                case "ToolResultBlock":
                    record_tool_result(block, trace, state)
                case _:
                    pass
        trace.flush()


def record_tool_result(block: Any, trace: Trace, state: ClaudeRecorderState) -> None:
    tid_raw = getattr(block, "tool_use_id", None)
    tid = str(tid_raw) if tid_raw is not None else ""
    pending = state.pending_tools.pop(tid, None)
    span_id = pending["span_id"] if pending else new_id()
    tool_name = pending["tool_name"] if pending else "unknown"
    args = pending["args"] if pending else {}
    start_time = pending["start_time"] if pending else datetime.now(UTC)
    end_time = datetime.now(UTC)
    is_err = bool(getattr(block, "is_error", False))

    trace.add_span(
        wire.tool_call_span(
            span_id=span_id,
            parent_span_id=trace.root_span_id,
            tool_name=tool_name,
            start_time=start_time,
            end_time=end_time,
            outcome="error" if is_err else "success",
        )
    )
    trace.add_event(
        wire.tool_call_start_event(
            event_id=new_id(),
            span_id=span_id,
            timestamp=start_time,
            args=args,
        )
    )
    trace.add_event(
        wire.tool_call_end_event(
            event_id=new_id(),
            span_id=span_id,
            timestamp=end_time,
            result=block.content,
            outcome="error" if is_err else "success",
            error=None,
        )
    )


def record_assistant_message(
    message: Any, trace: Trace, state: ClaudeRecorderState
) -> None:
    trace.begin_session()
    texts: list[str] = []

    for block in message.content:
        block_name = type(block).__name__
        match block_name:
            case "TextBlock":
                texts.append(block.text)
            case "ThinkingBlock":
                pass
            case "ToolUseBlock":
                tid_raw = getattr(block, "id", None)
                tid = str(tid_raw) if tid_raw is not None else str(new_id())
                state.pending_tools[tid] = {
                    "span_id": new_id(),
                    "tool_name": block.name,
                    "args": block.input,
                    "start_time": datetime.now(UTC),
                }
            case _:
                pass

    output = "\n".join(texts)
    if output.strip():
        now = datetime.now(UTC)
        model_sid = new_id()
        raw_model = getattr(message, "model", None)
        span_model = trace.model_name
        match raw_model:
            case str() if raw_model.strip():
                span_model = raw_model.strip()
            case _:
                pass
        trace.add_span(
            wire.model_call_span(
                span_id=model_sid,
                parent_span_id=trace.root_span_id,
                model_name=span_model,
                start_time=now,
                end_time=now,
                outcome="success",
            )
        )
        trace.add_event(
            wire.model_call_end_event(
                event_id=new_id(),
                span_id=model_sid,
                timestamp=now,
                output_payload=wire.text_redacted(output),
                outcome="success",
                error=None,
            )
        )
    trace.flush()


def record_message(message: Any, trace: Trace, state: ClaudeRecorderState) -> None:
    message_type = type(message).__name__

    match message_type:
        case "SystemMessage":
            data = getattr(message, "data", None) or {}
            trace.merge_system_message(data)
        case "UserMessage":
            record_user_message(message, trace, state)
        case "AssistantMessage":
            record_assistant_message(message, trace, state)
        case _:
            pass


CLAUDE_AGENT_SDK = "claude-agent-sdk"
