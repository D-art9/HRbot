from dataclasses import dataclass

from cyvia.client import CyviaClient


@dataclass(frozen=True, slots=True)
class WebToolAccessPolicy:
    agent_web_tool_access_control_enabled: bool
    agent_web_domain_allowlist: list[str]


def fetch_web_tool_access_policy(client: CyviaClient) -> WebToolAccessPolicy:
    response = client.get_json("/agent/web-tool-access-policy")
    response.raise_for_status()
    data = response.json()
    return WebToolAccessPolicy(
        agent_web_tool_access_control_enabled=bool(
            data["agent_web_tool_access_control_enabled"]
        ),
        agent_web_domain_allowlist=list(data["agent_web_domain_allowlist"]),
    )
