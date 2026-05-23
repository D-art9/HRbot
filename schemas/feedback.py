from typing import Optional

from pydantic import BaseModel


class Feedback(BaseModel):
    decision_id: str
    rating: int
    comment: Optional[str] = None
    timestamp: str
