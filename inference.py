import sys
import os
import json
import time

# Fix path so env/ folder is always found
sys.path.insert(0, os.getcwd())

from openai import OpenAI
from env.email_env import EmailTriageEnv
from env.models import Action

# ── Read environment variables ─────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.environ.get("MODEL_NAME")   or "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN     = os.environ.get("HF_TOKEN")     or "dummy-key"

# ── OpenAI client (required by submission rules) ───────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = """You are an email triage agent.
Respond ONLY with a JSON object, no markdown:
{
  "action_type": "label OR prioritize OR reply OR escalate OR archive OR done",
  "email_id": "the email id",
  "label": "urgent OR billing OR support OR spam OR hr OR legal OR null",
  "priority": 1 to 5 or null,
  "reply_body": "text or null",
  "escalate_to": "legal-team OR cto OR ceo OR null"
}"""

# ── Keyword fallback (used when API key is dummy) ──────────────
KEYWORDS = {
    "urgent":  ["urgent","down","outage","critical","crash","immediate","unresponsive"],
    "billing": ["invoice","charge","refund","billing","payment","cancel","subscription","charged"],
    "support": ["password","reset","cannot","help","error","issue","login","429","rate limit"],
    "spam":    ["winner","gift","prize","click here","congratulations","free webinar","selected"],
    "hr":      ["onboarding","hire","employee","hr","documents","sign","new hire"],
    "legal":   ["legal","patent","lawsuit","gdpr","breach","infringement","counsel","claim"],
}
PRIORITY = {"urgent":1,"legal":1,"billing":2,"hr":2,"support":3,"spam":5}
REPLIES  = {
    "billing": "Thank you for contacting us about your billing issue. Our team will resolve this within 1-2 business days.",
    "support": "Thank you for reaching out. Our support team will investigate and follow up within 24 hours.",
    "hr":      "Thank you. I have received the HR documents and will review them before the deadline.",
}
ESCALATE = {"urgent": "cto", "legal": "legal-team"}
ESCALATE_REASON = {
    "urgent": "Critical system issue requiring immediate CTO attention and emergency response.",
    "legal":  "Legal claim received requiring immediate review by legal team within deadline.",
}


def detect_label(email):
    text   = (email.subject + " " + email.body).lower()
    scores = {lb: sum(1 for kw in kws if kw in text) for lb, kws in KEYWORDS.items()}
    best   = max(scores, key=scores.get)
    return best if scores[best] > 0 else "support"


def rule_based_act(obs, labeled, prioritized, replied, escalated):
    """Fallback rule-based agent when no real API key is set."""
    for email in obs.inbox:
        if email.id not in labeled:
            label = detect_label(email)
            labeled.add(email.id)
            return Action(action_type="label", email_id=email.id, label=label)

    if obs.task_id in ("task_medium", "task_hard"):
        for email in obs.inbox:
            if email.id not in prioritized:
                label    = detect_label(email)
                priority = PRIORITY.get(label, 3)
                prioritized.add(email.id)
                return Action(action_type="prioritize", email_id=email.id, priority=priority)

    if obs.task_id == "task_hard":
        for email in obs.inbox:
            if email.id not in replied:
                label = detect_label(email)
                replied.add(email.id)
                if label in REPLIES:
                    return Action(action_type="reply", email_id=email.id, reply_body=REPLIES[label])

        for email in obs.inbox:
            if email.id not in escalated:
                label = detect_label(email)
                escalated.add(email.id)
                if label in ESCALATE:
                    return Action(
                        action_type="escalate",
                        email_id=email.id,
                        escalate_to=ESCALATE[label],
                        reply_body=ESCALATE_REASON[label],
                    )

    return Action(action_type="done", email_id=obs.inbox[0].id)


def llm_act(obs, history):
    """Call LLM via OpenAI client pointing to HuggingFace router."""
    emails_text = ""
    for e in obs.inbox:
        emails_text += f"[{e.id}] FROM: {e.sender}\nSUBJECT: {e.subject}\nBODY: {e.body}\n\n"

    user_msg = (
        f"TASK: {obs.instructions}\n"
        f"STEP: {obs.current_step}/{obs.max_steps}\n"
        f"SCORE SO FAR: {obs.score_so_far}\n\n"
        f"INBOX:\n{emails_text}\n"
        f"Choose the best single action. Output only JSON."
    )
    history.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        max_tokens=300,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    history.append({"role": "assistant", "content": raw})
    return Action(**json.loads(raw)), history


def run_task(task_id: str, seed: int = 42) -> dict:
    """
    Run one full episode.
    Emits [START], [STEP], and [END] structured logs.
    """
    env  = EmailTriageEnv(task_id=task_id, seed=seed)
    obs  = env.reset()

    use_llm     = HF_TOKEN not in ("dummy-key", "dummy-key-for-rule-based", "")
    history     = []
    labeled     = set()
    prioritized = set()
    replied     = set()
    escalated   = set()
    step        = 0

    # ── [START] log ────────────────────────────────────────────
    print(json.dumps({
        "event":     "START",
        "task_id":   task_id,
        "seed":      seed,
        "max_steps": obs.max_steps,
        "n_emails":  len(obs.inbox),
    }))

    start_time = time.time()

    while True:
        try:
            if use_llm:
                action, history = llm_act(obs, history)
            else:
                action = rule_based_act(obs, labeled, prioritized, replied, escalated)
        except Exception:
            action = rule_based_act(obs, labeled, prioritized, replied, escalated)

        result = env.step(action)

        # ── [STEP] log ─────────────────────────────────────────
        print(json.dumps({
            "event":       "STEP",
            "task_id":     task_id,
            "step":        result.observation.current_step,
            "action_type": action.action_type,
            "email_id":    action.email_id,
            "label":       action.label,
            "priority":    action.priority,
            "reward":      result.reward,
            "done":        result.done,
            "info":        result.info,
        }))

        obs   = result.observation
        step += 1

        if result.done:
            break

    episode = env.episode_result()
    elapsed = round(time.time() - start_time, 2)

    # ── [END] log ──────────────────────────────────────────────
    print(json.dumps({
        "event":            "END",
        "task_id":          task_id,
        "total_score":      episode.total_score,
        "steps_taken":      episode.steps_taken,
        "elapsed_seconds":  elapsed,
        "grader_breakdown": episode.grader_breakdown,
    }))

    return {
        "task_id":          episode.task_id,
        "total_score":      episode.total_score,
        "steps_taken":      episode.steps_taken,
        "grader_breakdown": episode.grader_breakdown,
    }


if __name__ == "__main__":
    print(json.dumps({
        "event":        "INFERENCE_START",
        "api_base_url": API_BASE_URL,
        "model_name":   MODEL_NAME,
        "tasks":        ["task_easy", "task_medium", "task_hard"],
    }))

    results     = {}
    total_start = time.time()

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        results[task_id] = run_task(task_id, seed=42)

    total_elapsed = round(time.time() - total_start, 2)
    all_valid     = all(0.0 <= r["total_score"] <= 1.0 for r in results.values())

    print(json.dumps({
        "event":         "INFERENCE_END",
        "total_elapsed": total_elapsed,
        "scores_valid":  all_valid,
        "results":       {k: v["total_score"] for k, v in results.items()},
    }))

    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({"event": "SAVED", "file": "inference_results.json"}))
    print("INFERENCE COMPLETE")