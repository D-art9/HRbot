from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from claude_agent_sdk.types import HookEvent, HookInput, HookMatcher, SyncHookJSONOutput

from cyvia.client import CyviaClient
from cyvia.web_tool_access_policy import (
    WebToolAccessPolicy,
    fetch_web_tool_access_policy,
)

if TYPE_CHECKING:
    from claude_agent_sdk.types import HookContext

WEB_SEARCH_TOOL_NAME = "WebSearch"
WEB_FETCH_TOOL_NAME = "WebFetch"
WEB_TOOLS_MATCHER = f"{WEB_SEARCH_TOOL_NAME}|{WEB_FETCH_TOOL_NAME}"


def build_claude_web_hooks(
    policy: WebToolAccessPolicy,
) -> dict[HookEvent, list[HookMatcher]]:
    """
    Return `ClaudeAgentOptions`-style hooks for tool access control.
    """
    if not policy.agent_web_tool_access_control_enabled:
        return {}

    tenant_allowlist = list(policy.agent_web_domain_allowlist)

    async def pre_tool_use_web_tools(
        hook_input: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        if hook_input["hook_event_name"] != "PreToolUse":
            return {}
        tool_name = hook_input["tool_name"]
        tool_input = hook_input["tool_input"]

        if tool_name == WEB_SEARCH_TOOL_NAME:
            updated_input = apply_tenant_web_search_allowed_domains(
                tenant_allowlist, tool_input
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "updatedInput": updated_input,
                }
            }

        if tool_name == WEB_FETCH_TOOL_NAME:
            raw_url = tool_input.get("url")
            if not isinstance(raw_url, str) or not raw_url.strip():
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "WebFetch url is missing or not a string.",
                    }
                }
            if not web_fetch_url_allowed(raw_url, tenant_allowlist):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            "WebFetch URL host is not allowed by the tenant domain allowlist."
                        ),
                    }
                }
            return {}

        return {}

    matcher = HookMatcher(
        matcher=WEB_TOOLS_MATCHER,
        hooks=[pre_tool_use_web_tools],
    )
    return {"PreToolUse": [matcher]}


def apply_tenant_web_search_allowed_domains(
    tenant_allowlist: list[str],
    tool_input: dict[str, Any],
) -> dict[str, Any]:
    # WebSearch tool_input shape: https://code.claude.com/docs/en/agent-sdk/python#websearch
    out: dict[str, Any] = dict(tool_input)
    out["allowed_domains"] = list(tenant_allowlist)

    if "blocked_domains" in out and out["blocked_domains"] is not None:
        out.pop("blocked_domains")
        warnings.warn(
            "Cyvia removed WebSearch blocked_domains because allowed_domains"
            " and blocked_domains cannot both be set; the tenant allowlist is"
            " applied as allowed_domains.",
            UserWarning,
            stacklevel=2,
        )

    return out


def web_fetch_url_allowed(url: str, tenant_allowlist: list[str]) -> bool:
    parsed = urlparse(url.strip())
    host = parsed.hostname
    if host is None:
        return False
    return is_hostname_allowed_by_allowlist(host.lower(), tenant_allowlist)


def is_hostname_allowed_by_allowlist(host: str, tenant_allowlist: list[str]) -> bool:
    if not tenant_allowlist:
        return False
    for domain in tenant_allowlist:
        allowed = domain.strip().lower()
        if not allowed:
            continue
        if host == allowed or host.endswith("." + allowed):
            return True
    return False


def fetch_claude_web_tool_hooks(
    client: CyviaClient,
) -> dict[HookEvent, list[HookMatcher]]:
    """Fetch tenant web tool access policy from the API and build PreToolUse hooks."""
    return build_claude_web_hooks(fetch_web_tool_access_policy(client))


def merge_claude_hooks(
    user_hooks: dict[HookEvent, list[HookMatcher]] | None,
    cyvia_hooks: dict[HookEvent, list[HookMatcher]] | None,
) -> dict[HookEvent, list[HookMatcher]]:
    """Merge hook maps for `ClaudeAgentOptions`: user hooks first, then Cyvia (enforcement runs after user hooks)."""
    user_hooks = user_hooks or {}
    cyvia_hooks = cyvia_hooks or {}
    out: dict[HookEvent, list[HookMatcher]] = {}
    for event in set(user_hooks) | set(cyvia_hooks):
        chain = user_hooks.get(event, []) + cyvia_hooks.get(event, [])
        if chain:
            out[event] = chain
    return out
