from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult

from .. import wire
from ..trace import Trace, new_id

LANGCHAIN = "langchain"


@dataclass
class _PendingModel:
    span_id: UUID
    start_time: datetime
    model_name: str
    parent_span_id: UUID
    input_for_start_event: str


@dataclass
class _PendingTool:
    span_id: UUID
    tool_name: str
    start_time: datetime
    parent_span_id: UUID
    args: Any


class LangChainCallbackHandler(BaseCallbackHandler):
    def __init__(self, trace: Trace) -> None:
        super().__init__()
        self.trace = trace
        self.run_to_span: dict[str, UUID] = {}
        self.pending_model: dict[str, _PendingModel] = {}
        self.pending_tool: dict[str, _PendingTool] = {}
        self._previous_chat_messages: list[BaseMessage] | None = None

    def _run_key(self, run_id: UUID) -> str:
        return str(run_id)

    def _resolve_parent_span_id(self, parent_run_id: UUID | None) -> UUID:
        if parent_run_id is None:
            return self.trace.root_span_id
        return self.run_to_span.get(
            self._run_key(parent_run_id), self.trace.root_span_id
        )

    def _ensure_session(self) -> None:
        self.trace.begin_session()

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is not None:
            return None
        self._ensure_session()
        emit_input_events_from_mapping(self.trace, inputs)
        return None

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._ensure_session()
        flat = messages[0] if messages else []
        suffix = chat_messages_new_suffix(self._previous_chat_messages, flat)
        self._previous_chat_messages = list(flat)
        emit_message_role_events(self.trace, suffix)
        rk = self._run_key(run_id)
        span_id = new_id()
        self.run_to_span[rk] = span_id
        self.pending_model[rk] = _PendingModel(
            span_id=span_id,
            start_time=datetime.now(UTC),
            model_name=model_name_from_serialized(serialized),
            parent_span_id=self._resolve_parent_span_id(parent_run_id),
            input_for_start_event=messages_to_plain_input(flat),
        )
        return None

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        rk = self._run_key(run_id)
        if rk in self.pending_model:
            return None
        self._ensure_session()
        joined = "\n".join(prompts)
        if joined.strip():
            self.trace.add_event(
                wire.prompt_received_event(
                    event_id=new_id(),
                    span_id=self.trace.root_span_id,
                    timestamp=datetime.now(UTC),
                    input_payload=wire.text_redacted(joined),
                )
            )
            self.trace.flush()
        span_id = new_id()
        self.run_to_span[rk] = span_id
        self.pending_model[rk] = _PendingModel(
            span_id=span_id,
            start_time=datetime.now(UTC),
            model_name=model_name_from_serialized(serialized),
            parent_span_id=self._resolve_parent_span_id(parent_run_id),
            input_for_start_event=joined,
        )
        return None

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        rk = self._run_key(run_id)
        pending = self.pending_model.pop(rk, None)
        self.run_to_span.pop(rk, None)
        if pending is None:
            return None
        end = datetime.now(UTC)
        text = text_from_llm_result(response)
        finalize_model_span(
            self.trace,
            pending=pending,
            end_time=end,
            outcome="success",
            output_text=text,
            error=None,
        )
        return None

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        rk = self._run_key(run_id)
        pending = self.pending_model.pop(rk, None)
        self.run_to_span.pop(rk, None)
        if pending is None:
            return None
        end = datetime.now(UTC)
        finalize_model_span(
            self.trace,
            pending=pending,
            end_time=end,
            outcome="error",
            output_text="",
            error=exception_to_error_payload(error),
        )
        return None

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._ensure_session()
        rk = self._run_key(run_id)
        span_id = new_id()
        self.run_to_span[rk] = span_id
        args: Any = (
            inputs if inputs is not None else tool_args_from_input_str(input_str)
        )
        self.pending_tool[rk] = _PendingTool(
            span_id=span_id,
            tool_name=tool_name_from_serialized(serialized),
            start_time=datetime.now(UTC),
            parent_span_id=self._resolve_parent_span_id(parent_run_id),
            args=args,
        )
        return None

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        rk = self._run_key(run_id)
        pending = self.pending_tool.pop(rk, None)
        self.run_to_span.pop(rk, None)
        if pending is None:
            return None
        end = datetime.now(UTC)
        finalize_tool_span(
            self.trace,
            pending=pending,
            end_time=end,
            result=output,
            outcome="success",
            error=None,
        )
        return None

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        rk = self._run_key(run_id)
        pending = self.pending_tool.pop(rk, None)
        self.run_to_span.pop(rk, None)
        if pending is None:
            return None
        end = datetime.now(UTC)
        finalize_tool_span(
            self.trace,
            pending=pending,
            end_time=end,
            result=None,
            outcome="error",
            error=exception_to_error_payload(error),
        )
        return None


def finalize_model_span(
    trace: Trace,
    *,
    pending: _PendingModel,
    end_time: datetime,
    outcome: str,
    output_text: str,
    error: dict[str, Any] | None,
) -> None:
    trace.add_span(
        wire.model_call_span(
            span_id=pending.span_id,
            parent_span_id=pending.parent_span_id,
            model_name=pending.model_name,
            start_time=pending.start_time,
            end_time=end_time,
            outcome=outcome,
        )
    )
    trace.add_event(
        wire.model_call_start_event(
            event_id=new_id(),
            span_id=pending.span_id,
            timestamp=pending.start_time,
            input_payload=wire.text_redacted(pending.input_for_start_event),
        )
    )
    trace.add_event(
        wire.model_call_end_event(
            event_id=new_id(),
            span_id=pending.span_id,
            timestamp=end_time,
            output_payload=wire.text_redacted(output_text),
            outcome=outcome,
            error=error,
        )
    )
    trace.flush()


def finalize_tool_span(
    trace: Trace,
    *,
    pending: _PendingTool,
    end_time: datetime,
    result: Any,
    outcome: str,
    error: dict[str, Any] | None,
) -> None:
    trace.add_span(
        wire.tool_call_span(
            span_id=pending.span_id,
            parent_span_id=pending.parent_span_id,
            tool_name=pending.tool_name,
            start_time=pending.start_time,
            end_time=end_time,
            outcome=outcome,
        )
    )
    trace.add_event(
        wire.tool_call_start_event(
            event_id=new_id(),
            span_id=pending.span_id,
            timestamp=pending.start_time,
            args=pending.args,
        )
    )
    trace.add_event(
        wire.tool_call_end_event(
            event_id=new_id(),
            span_id=pending.span_id,
            timestamp=end_time,
            result=tool_result_for_wire(result),
            outcome=outcome,
            error=error,
        )
    )
    trace.flush()


def tool_result_for_wire(output: Any) -> Any:
    from langchain_core.messages import BaseMessage

    if isinstance(output, BaseMessage):
        return message_content_to_text(output.content)
    if isinstance(output, (str, int, float, bool)) or output is None:
        return output
    if isinstance(output, dict):
        return {str(k): tool_result_for_wire(v) for k, v in output.items()}
    if isinstance(output, (list, tuple)):
        return [tool_result_for_wire(x) for x in output]
    return str(output)


def emit_input_events_from_mapping(trace: Trace, inputs: dict[str, Any]) -> None:
    msgs = inputs.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], BaseMessage):
        return
    any_ev = False
    for v in inputs.values():
        if isinstance(v, BaseMessage):
            any_ev = _emit_single_message_event(trace, v) or any_ev
        elif isinstance(v, list) and v and isinstance(v[0], BaseMessage):
            for m in v:
                any_ev = _emit_single_message_event(trace, m) or any_ev
        elif isinstance(v, str) and v.strip():
            trace.add_event(
                wire.prompt_received_event(
                    event_id=new_id(),
                    span_id=trace.root_span_id,
                    timestamp=datetime.now(UTC),
                    input_payload=wire.text_redacted(v),
                )
            )
            any_ev = True
    if any_ev:
        trace.flush()


def emit_message_role_events(trace: Trace, messages: list[BaseMessage]) -> None:
    any_ev = False
    for m in messages:
        any_ev = _emit_single_message_event(trace, m) or any_ev
    if any_ev:
        trace.flush()


def _emit_single_message_event(trace: Trace, m: BaseMessage) -> bool:
    now = datetime.now(UTC)
    if isinstance(m, SystemMessage):
        txt = message_content_to_text(m.content)
        if not txt.strip():
            return False
        trace.add_event(
            wire.system_instructions_loaded_event(
                event_id=new_id(),
                span_id=trace.root_span_id,
                timestamp=now,
                instructions=wire.text_redacted(txt),
            )
        )
        return True
    if isinstance(m, HumanMessage):
        txt = message_content_to_text(m.content)
        if not txt.strip():
            return False
        trace.add_event(
            wire.prompt_received_event(
                event_id=new_id(),
                span_id=trace.root_span_id,
                timestamp=now,
                input_payload=wire.text_redacted(txt),
            )
        )
        return True
    return False


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def messages_equal_for_history(a: BaseMessage, b: BaseMessage) -> bool:
    if type(a) is not type(b):
        return False
    return message_content_to_text(a.content) == message_content_to_text(b.content)


def chat_messages_new_suffix(
    previous: list[BaseMessage] | None, current: list[BaseMessage]
) -> list[BaseMessage]:
    if not previous:
        return list(current)
    n_match = 0
    limit = min(len(previous), len(current))
    while n_match < limit and messages_equal_for_history(
        previous[n_match], current[n_match]
    ):
        n_match += 1
    return list(current[n_match:])


def messages_to_plain_input(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for m in messages:
        role = getattr(m, "type", type(m).__name__)
        lines.append(f"{role}: {message_content_to_text(m.content)}")
    return "\n".join(lines)


def model_name_from_serialized(serialized: dict[str, Any]) -> str:
    name = serialized.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    id_path = serialized.get("id")
    if isinstance(id_path, list) and id_path:
        last = id_path[-1]
        if isinstance(last, str):
            return last
    return "unknown"


def text_from_llm_result(response: LLMResult) -> str:
    texts: list[str] = []
    for gen_list in response.generations:
        for gen in gen_list:
            if getattr(gen, "text", None):
                texts.append(str(gen.text))
            elif getattr(gen, "message", None) is not None:
                texts.append(message_content_to_text(gen.message.content))
    return "\n".join(texts).strip()


def tool_name_from_serialized(serialized: dict[str, Any]) -> str:
    n = serialized.get("name")
    if isinstance(n, str) and n:
        return n
    return "unknown_tool"


def tool_args_from_input_str(input_str: str) -> Any:
    try:
        return json.loads(input_str)
    except json.JSONDecodeError:
        return {"input": input_str}


def exception_to_error_payload(exc: BaseException) -> dict[str, Any]:
    return {"message": wire.text_redacted(str(exc))}
