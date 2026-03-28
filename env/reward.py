from __future__ import annotations
from .models import Action

LABEL_WEIGHTS    = {"task_easy": 1.0, "task_medium": 0.5, "task_hard": 0.25}
PRIORITY_WEIGHTS = {"task_medium": 0.5, "task_hard": 0.20}
REPLY_WEIGHT     = 0.35
ESCALATE_WEIGHT  = 0.20


class RewardCalculator:
    def __init__(self, task_id: str):
        self.task_id = task_id

    def score_action(self, action: Action, emails_map: dict, agent_state: dict):
        eid = action.email_id
        if eid not in emails_map:
            return -0.05, {"error": "unknown email_id"}

        email  = emails_map[eid]
        reward = 0.0
        info   = {}

        if action.action_type == "label" and action.label:
            correct = email.get("_true_label")
            if action.label == correct:
                reward = LABEL_WEIGHTS.get(self.task_id, 0.25)
                info["result"] = "correct label"
            else:
                reward = -0.05
                info["result"] = f"wrong label, expected {correct}"

        elif action.action_type == "prioritize" and action.priority is not None:
            correct_p = email.get("_true_priority", 3)
            diff      = abs(action.priority - correct_p)
            w         = PRIORITY_WEIGHTS.get(self.task_id, 0.0)
            reward    = w * max(0.0, 1.0 - diff * 0.25)
            info["priority_diff"] = diff

        elif action.action_type == "reply" and self.task_id == "task_hard":
            if email.get("_requires_reply") and action.reply_body and len(action.reply_body) > 40:
                reward = REPLY_WEIGHT / 10
                info["result"] = "reply accepted"
            elif not email.get("_requires_reply"):
                reward = -0.02
                info["result"] = "unnecessary reply"

        elif action.action_type == "escalate" and self.task_id == "task_hard":
            has_reason = bool(action.reply_body and len(action.reply_body) > 20)
            if email.get("_requires_escalation") and action.escalate_to and has_reason:
                reward = ESCALATE_WEIGHT / 10
                info["result"] = "escalation accepted"
            elif not email.get("_requires_escalation"):
                reward = -0.03
                info["result"] = "unnecessary escalation"

        return round(reward, 4), info