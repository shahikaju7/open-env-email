from __future__ import annotations
from env.models import Observation, Action


# ── Keyword lookup tables ────────────────────────────────────
# These are the words the agent searches for in subject + body

URGENT_KEYWORDS   = ["urgent", "down", "outage", "critical", "immediate",
                     "emergency", "unresponsive", "broken", "crash"]

BILLING_KEYWORDS  = ["invoice", "charge", "refund", "payment", "billing",
                     "subscription", "credit", "cancel", "charged twice",
                     "cancel our", "cancellation"]

SUPPORT_KEYWORDS  = ["password", "reset", "cannot", "help", "issue",
                     "error", "problem", "not working", "broken",
                     "login", "access", "429", "rate limit", "api"]

SPAM_KEYWORDS     = ["winner", "gift card", "prize", "congratulations",
                     "click here", "free webinar", "limited seats",
                     "claim your", "selected", "invite"]

HR_KEYWORDS       = ["onboarding", "hire", "employee", "hr", "sign",
                     "documents", "new hire", "human resources"]

LEGAL_KEYWORDS    = ["legal", "patent", "lawsuit", "gdpr", "compliance",
                     "infringement", "counsel", "breach", "attorney",
                     "court", "claim"]

# Priority mapping — which labels get which urgency score
LABEL_TO_PRIORITY = {
    "urgent":  1,
    "legal":   1,
    "billing": 2,
    "hr":      2,
    "support": 3,
    "spam":    5,
}

# Reply templates for each label
REPLY_TEMPLATES = {
    "billing": (
        "Thank you for reaching out about your billing concern. "
        "I have reviewed your account and our billing team will process "
        "your request within 1-2 business days. "
        "Please let us know if you need any further assistance."
    ),
    "support": (
        "Thank you for contacting our support team. "
        "I have reviewed your issue and our technical team will investigate "
        "and follow up with you within 24 hours. "
        "We apologize for any inconvenience caused."
    ),
    "hr": (
        "Thank you for your message. "
        "I have received the HR documents and will review and return them "
        "before the deadline. Please let me know if anything else is needed."
    ),
}

# Escalation targets
ESCALATION_TARGETS = {
    "urgent": "cto",
    "legal":  "legal-team",
}

ESCALATION_REASONS = {
    "urgent": (
        "This email reports a critical system outage affecting customers. "
        "Immediate escalation to CTO required for emergency response."
    ),
    "legal": (
        "This email contains legal claims or compliance concerns. "
        "Escalating to legal team for immediate review and response."
    ),
}


def detect_label(email) -> str:
    """
    Scan subject + body for keywords.
    Returns the best matching label.
    """
    # Combine subject and body, make lowercase for easy matching
    text = (email.subject + " " + email.body).lower()

    # Count how many keywords match for each label
    scores = {
        "urgent":  sum(1 for kw in URGENT_KEYWORDS  if kw in text),
        "billing": sum(1 for kw in BILLING_KEYWORDS if kw in text),
        "support": sum(1 for kw in SUPPORT_KEYWORDS if kw in text),
        "spam":    sum(1 for kw in SPAM_KEYWORDS    if kw in text),
        "hr":      sum(1 for kw in HR_KEYWORDS      if kw in text),
        "legal":   sum(1 for kw in LEGAL_KEYWORDS   if kw in text),
    }

    # Pick the label with the highest keyword score
    best_label = max(scores, key=scores.get)

    # If no keywords matched at all, default to support
    if scores[best_label] == 0:
        return "support"

    return best_label


class RuleBasedAgent:
    """
    A simple agent that uses keyword matching to triage emails.
    No API. No internet. No cost. Works completely offline.

    Strategy per task:
      task_easy   — just label every email
      task_medium — label + prioritize every email
      task_hard   — label + prioritize + reply + escalate
    """

    def __init__(self, task_id: str = "task_easy"):
        self.task_id = task_id
        # Track what we have done to each email
        self._labeled:    set = set()
        self._prioritized: set = set()
        self._replied:    set = set()
        self._escalated:  set = set()

    def act(self, obs: Observation) -> Action:
        """
        Look at the current observation and decide one action.
        We process emails one at a time, finishing all of one
        action type before moving to the next.
        """

        # Step 1 — Find the next email that still needs a label
        for email in obs.inbox:
            if email.id not in self._labeled:
                label = detect_label(email)
                self._labeled.add(email.id)
                return Action(
                    action_type="label",
                    email_id=email.id,
                    label=label,
                )

        # All emails are now labeled.
        # Step 2 (medium + hard) — Prioritize every email
        if self.task_id in ("task_medium", "task_hard"):
            for email in obs.inbox:
                if email.id not in self._prioritized:
                    label    = detect_label(email)
                    priority = LABEL_TO_PRIORITY.get(label, 3)
                    self._prioritized.add(email.id)
                    return Action(
                        action_type="prioritize",
                        email_id=email.id,
                        priority=priority,
                    )

        # Step 3 (hard only) — Reply to emails that need a reply
        if self.task_id == "task_hard":
            for email in obs.inbox:
                if email.id not in self._replied:
                    label = detect_label(email)
                    if label in REPLY_TEMPLATES:
                        reply_text = REPLY_TEMPLATES[label]
                        self._replied.add(email.id)
                        return Action(
                            action_type="reply",
                            email_id=email.id,
                            reply_body=reply_text,
                        )
                    else:
                        # No reply needed for this label — mark as done
                        self._replied.add(email.id)

        # Step 4 (hard only) — Escalate urgent and legal emails
        if self.task_id == "task_hard":
            for email in obs.inbox:
                if email.id not in self._escalated:
                    label = detect_label(email)
                    if label in ESCALATION_TARGETS:
                        target = ESCALATION_TARGETS[label]
                        reason = ESCALATION_REASONS[label]
                        self._escalated.add(email.id)
                        return Action(
                            action_type="escalate",
                            email_id=email.id,
                            escalate_to=target,
                            reply_body=reason,
                        )
                    else:
                        self._escalated.add(email.id)

        # All done — signal end of episode
        return Action(action_type="done", email_id=obs.inbox[0].id)