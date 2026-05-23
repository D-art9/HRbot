from typing import Any, Dict

from pydantic import BaseModel


class DecisionLog(BaseModel):
    module: str
    query: str
    response: str
    reasoning: str
    metadata: Dict[str, Any]
