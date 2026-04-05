code = '''
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from env.email_env import EmailTriageEnv
from env.models import Action

# ── Environment variables ──────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "meta-llama/Llama-3.1-8B-Instruct"
API_KEY      = os.getenv("HF_TOKEN")     or "dummy-key"

TEMPERATURE      = 0.1
MAX_TOKENS       = 300
MAX_STEPS        = 40
MAX_TOTAL_REWARD = 10.0
SUCCESS_SCORE_THRESHOLD = 0.5
TASK_NAME        = "email-triage"
BENCHMARK        = "EmailTriageEnv"

# ── OpenAI client ──────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Structured log functions (exact format required) ───────────
def log_start(task, env, model):
    print(json.dumps({
        "event":     "START",
        "task":      task,
        "env":       env,
        "model":     model,
        "timestamp": time.time(),
    }), flush=True)

def log_step(step, action, reward, done, error=None):
    print(json.dumps({
        "event":   "STEP",
        "step":    step,
        "action":  action,
        "reward":  reward,
        "done":    done,
        "error":   error,
    }), flush=True)

def log_end(task_id, success, steps, score, rewards):
    print(json.dumps({
        "event":   "END",
        "task_id": task_id,
        "success": success,
        "steps":   steps,
        "score":   score,
        "rewards": rewards,
    }), flush=True)

# ── System prompt ──────────────────────────────────────────────
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

# ── Keyword fallback ───────────────────────────────────────────
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
ESCALATE = {"urgent":"cto","legal":"legal-team"}
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

def get_model_message(obs, step, history):
    """Call LLM via OpenAI client."""
    emails_text = ""
    for e in obs.inbox:
        emails_text += f"[{e.id}] FROM: {e.sender}\\nSUBJECT: {e.subject}\\nBODY: {e.body}\\n\\n"
    user_prompt = (
        f"TASK: {obs.instructions}\\n"
        f"STEP: {step}/{MAX_STEPS}\\n"
        f"INBOX:\\n{emails_text}\\n"
        f"Output only JSON."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "done"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "done"

def run_task(task_id: str, seed: int = 42) -> dict:
    env  = EmailTriageEnv(task_id=task_id, seed=seed)
    obs  = env.reset()

    use_llm     = API_KEY not in ("dummy-key", "dummy-key-for-rule-based", "")
    history     = []
    rewards     = []
    labeled     = set()
    prioritized = set()
    replied     = set()
    escalated   = set()
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            # Decide action
            try:
                if use_llm:
                    raw = get_model_message(obs, step, history)
                    raw = raw.replace("```json"," ").replace("```"," ").strip()
                    action = Action(**json.loads(raw))
                else:
                    action = rule_based_act(obs, labeled, prioritized, replied, escalated)
            except Exception:
                action = rule_based_act(obs, labeled, prioritized, replied, escalated)

            # Apply action
            result  = env.step(action)
            reward  = result.reward or 0.0
            done    = result.done
            error   = None

            rewards.append(reward)
            steps_taken = step
            obs = result.observation

            log_step(
                step=step,
                action=action.model_dump(),
                reward=reward,
                done=done,
                error=error,
            )

            history.append(f"Step {step}: {action.action_type} -> reward {reward:+.2f}")

            if done:
                break

        score   = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score   = min(max(score, 0.0), 1.0)
        ep      = env.episode_result()
        score   = ep.total_score
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(task_id=task_id, success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id":          task_id,
        "total_score":      score,
        "steps_taken":      steps_taken,
        "grader_breakdown": ep.grader_breakdown,
    }

if __name__ == "__main__":
    print(json.dumps({
        "event":        "INFERENCE_START",
        "api_base_url": API_BASE_URL,
        "model_name":   MODEL_NAME,
        "tasks":        ["task_easy", "task_medium", "task_hard"],
    }), flush=True)

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
    }), flush=True)

    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({"event": "SAVED", "file": "inference_results.json"}), flush=True)
'''