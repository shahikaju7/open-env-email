import os
import json
from openai import OpenAI
from env.email_env import EmailTriageEnv
from env.models import Action

# ── Read environment variables (set these in HuggingFace Secrets) ──
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-3.5-turbo")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

# ── OpenAI client (required by submission rules) ──────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy-key-for-rule-based",
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


def agent_act(obs, history):
    """Use OpenAI client to decide action. Falls back to rule-based if no key."""
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

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            max_tokens=300,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": raw})
        # Clean any markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        return Action(**json.loads(raw)), history

    except Exception as e:
        # Fallback to rule-based if API unavailable
        return rule_based_fallback(obs), history


def rule_based_fallback(obs):
    """Simple keyword fallback when API is not available."""
    KEYWORDS = {
        "urgent":  ["urgent","down","outage","critical","crash","immediate"],
        "billing": ["invoice","charge","refund","billing","payment","cancel","subscription"],
        "support": ["password","reset","cannot","help","error","issue","login","429"],
        "spam":    ["winner","gift","prize","click here","congratulations","free webinar"],
        "hr":      ["onboarding","hire","employee","hr","documents","sign"],
        "legal":   ["legal","patent","lawsuit","gdpr","breach","infringement","counsel"],
    }
    PRIORITY  = {"urgent":1,"legal":1,"billing":2,"hr":2,"support":3,"spam":5}
    ESCALATE  = {"urgent":"cto","legal":"legal-team"}
    REPLIES   = {
        "billing": "Thank you for contacting us about your billing issue. Our team will resolve this within 1-2 business days.",
        "support": "Thank you for reaching out. Our support team will investigate and follow up within 24 hours.",
        "hr":      "Thank you. I have received the HR documents and will review them before the deadline.",
    }

    for e in obs.inbox:
        text   = (e.subject + " " + e.body).lower()
        scores = {lb: sum(1 for kw in kws if kw in text) for lb, kws in KEYWORDS.items()}
        label  = max(scores, key=scores.get) if max(scores.values()) > 0 else "support"

        # Check what still needs doing
        # We track state via obs — pick first incomplete action
        return Action(action_type="label", email_id=e.id, label=label)

    return Action(action_type="done", email_id=obs.inbox[0].id)


def run_task(task_id: str, seed: int = 42) -> dict:
    """Run one full episode and return the graded result."""
    print(f"\n  Running {task_id}...")
    env     = EmailTriageEnv(task_id=task_id, seed=seed)
    obs     = env.reset()
    history = []
    step    = 0

    while True:
        action, history = agent_act(obs, history)
        result = env.step(action)
        print(f"    step {step+1:2d}: {action.action_type:12s} [{action.email_id}]  reward={result.reward:+.4f}")
        obs   = result.observation
        step += 1
        if result.done:
            break

    ep = env.episode_result()
    print(f"  Score: {ep.total_score:.4f}")
    return {
        "task_id":          ep.task_id,
        "total_score":      ep.total_score,
        "steps_taken":      ep.steps_taken,
        "grader_breakdown": ep.grader_breakdown,
    }


if __name__ == "__main__":
    print("=" * 55)
    print("  EmailTriageEnv — Inference Script")
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print("=" * 55)

    results = {}
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        results[task_id] = run_task(task_id, seed=42)

    print("\n" + "=" * 55)
    print("  FINAL SCORES")
    print("=" * 55)
    for task_id, r in results.items():
        bar = "█" * int(r["total_score"] * 20) + "░" * (20 - int(r["total_score"] * 20))
        print(f"  {task_id:15s}  {r['total_score']:.4f}  {bar}")

    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to inference_results.json")

    # Validate all scores are in 0.0-1.0 range
    all_valid = all(0.0 <= r["total_score"] <= 1.0 for r in results.values())
    print(f"\nScore range valid (0.0-1.0): {all_valid}")
    print("INFERENCE COMPLETE")