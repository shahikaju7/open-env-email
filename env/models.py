from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Email(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str
    thread_id: Optional[str] = None


class Observation(BaseModel):
    inbox: list[Email]
    current_step: int
    max_steps: int
    task_id: str
    instructions: str
    score_so_far: float = 0.0


class Action(BaseModel):
    action_type: Literal["label", "prioritize", "reply", "escalate", "archive", "done"]
    email_id: str
    label: Optional[Literal["urgent", "billing", "support", "spam", "hr", "legal"]] = None
    priority: Optional[int] = Field(default=None, ge=1, le=5)
    reply_body: Optional[str] = None
    escalate_to: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


class EpisodeResult(BaseModel):
    task_id: str
    total_score: float
    steps_taken: int
    actions_log: list[dict]
    grader_breakdown: dict