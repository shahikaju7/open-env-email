from __future__ import annotations
import json
from pathlib import Path
from .models import Action, Observation, StepResult, EpisodeResult, Email
from .reward import RewardCalculator
from .graders import GRADERS

DATA_DIR = Path(__file__).parent / "data"

TASK_CONFIGS = {
    "task_easy":   {"max_steps": 15, "n_emails": 10},
    "task_medium": {"max_steps": 25, "n_emails": 10},
    "task_hard":   {"max_steps": 40, "n_emails": 10},
}

INSTRUCTIONS = {
    "task_easy":   "Label each email: urgent | billing | support | spam | hr | legal. Use action_type=label. Call done when finished.",
    "task_medium": "Label each email AND set priority 1-5. Use separate label and prioritize actions. Call done when all emails have both.",
    "task_hard":   "Label and prioritize every email. Reply to support/billing/hr emails. Escalate urgent/legal emails. Call done when finished.",
}


class EmailTriageEnv:

    def __init__(self, task_id: str = "task_easy", seed: int = 42):
        assert task_id in TASK_CONFIGS
        self.task_id      = task_id
        self.seed         = seed
        self._cfg         = TASK_CONFIGS[task_id]
        self._emails_raw  = json.loads((DATA_DIR / "emails.json").read_text())
        self._grader      = GRADERS[task_id]
        self._reward_calc = RewardCalculator(task_id)
        self._reset_state()

    def reset(self) -> Observation:
        self._reset_state()
        return self._make_obs()

    def step(self, action: Action | dict) -> StepResult:
        if isinstance(action, dict):
            action = Action(**action)
        self._current_step += 1
        self._actions_log.append(action.model_dump())
        reward, info = self._reward_calc.score_action(action, self._emails_map, self._agent_state)
        self._cumulative_reward += reward
        self._update_agent_state(action)
        done = action.action_type == "done" or self._current_step >= self._cfg["max_steps"]
        return StepResult(observation=self._make_obs(), reward=reward, done=done, info=info)

    def state(self) -> dict:
        return {
            "task_id":      self.task_id,
            "current_step": self._current_step,
            "agent_state":  self._agent_state,
        }

    def episode_result(self) -> EpisodeResult:
        inbox     = [Email(**{k: v for k, v in e.items() if not k.startswith("_")})
                     for e in self._emails_map.values()]
        breakdown = self._grader.grade(inbox, self._agent_state, self._actions_log)
        return EpisodeResult(
            task_id=self.task_id,
            total_score=breakdown["total_score"],
            steps_taken=self._current_step,
            actions_log=self._actions_log,
            grader_breakdown=breakdown,
        )

    def _reset_state(self):
        import random
        rng    = random.Random(self.seed)
        sample = rng.sample(self._emails_raw, self._cfg["n_emails"])
        self._emails_map        = {e["id"]: e for e in sample}
        self._current_step      = 0
        self._cumulative_reward = 0.0
        self._actions_log       = []
        self._agent_state       = {
            eid: {"label": None, "priority": None, "reply": None, "escalated": False}
            for eid in self._emails_map
        }

    def _make_obs(self) -> Observation:
        safe = ["id", "subject", "sender", "body", "timestamp", "thread_id"]
        return Observation(
            inbox=[Email(**{k: e[k] for k in safe if k in e}) for e in self._emails_map.values()],
            current_step=self._current_step,
            max_steps=self._cfg["max_steps"],
            task_id=self.task_id,
            instructions=INSTRUCTIONS[self.task_id],
            score_so_far=round(self._cumulative_reward, 4),
        )

    def _update_agent_state(self, action: Action):
        eid = action.email_id
        if eid not in self._agent_state:
            return
        s = self._agent_state[eid]
        if action.action_type == "label"      and action.label:      s["label"]    = action.label
        if action.action_type == "prioritize" and action.priority:   s["priority"] = action.priority
        if action.action_type == "reply"      and action.reply_body: s["reply"]    = action.reply_body
        if action.action_type == "escalate":
            s["escalated"] = True
            s["reply"]     = action.reply_body