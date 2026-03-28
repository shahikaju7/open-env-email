from __future__ import annotations


class EasyGrader:
    def grade(self, emails, agent_state, actions_log):
        labeled = sum(1 for e in emails if agent_state.get(e.id, {}).get("label"))
        total   = len(emails)
        return {
            "total_score":    round(labeled / max(total, 1), 4),
            "emails_labeled": labeled,
            "total_emails":   total,
        }


class MediumGrader:
    def grade(self, emails, agent_state, actions_log):
        label_scores    = [1.0 if agent_state.get(e.id, {}).get("label")    else 0.0 for e in emails]
        priority_scores = [1.0 if agent_state.get(e.id, {}).get("priority") else 0.0 for e in emails]
        la = sum(label_scores)    / len(label_scores)
        pa = sum(priority_scores) / len(priority_scores)
        return {
            "total_score":    round(0.5 * la + 0.5 * pa, 4),
            "label_score":    round(la, 4),
            "priority_score": round(pa, 4),
        }


class HardGrader:
    def grade(self, emails, agent_state, actions_log):
        ls = [1.0 if agent_state.get(e.id, {}).get("label")    else 0.0 for e in emails]
        ps = [1.0 if agent_state.get(e.id, {}).get("priority") else 0.0 for e in emails]
        rs = [min(1.0, len(agent_state.get(e.id, {}).get("reply") or "") / 150) for e in emails]
        es = [1.0 if agent_state.get(e.id, {}).get("escalated") else 0.0 for e in emails]

        def avg(x): return sum(x) / len(x) if x else 0.0

        total = 0.25*avg(ls) + 0.20*avg(ps) + 0.35*avg(rs) + 0.20*avg(es)
        return {
            "total_score":      round(total, 4),
            "label_score":      round(avg(ls), 4),
            "priority_score":   round(avg(ps), 4),
            "reply_score":      round(avg(rs), 4),
            "escalation_score": round(avg(es), 4),
        }


GRADERS = {
    "task_easy":   EasyGrader(),
    "task_medium": MediumGrader(),
    "task_hard":   HardGrader(),
}