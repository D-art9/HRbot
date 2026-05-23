from typing import Optional

from pydantic import BaseModel


class CandidateProfile(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    experience: Optional[int] = None
    salary_expectation: Optional[str] = None
    joining_date: Optional[str] = None
