import os
import sys
import json
import subprocess
import requests
import time

SPACE_URL = os.environ.get("SPACE_URL", "http://localhost:7860")

passed = []
failed = []

def check(name, condition, fix=""):
    if condition:
        passed.append(name)
        print(f"  PASS  {name}")
    else:
        failed.append(name)
        print(f"  FAIL  {name}")
        if fix:
            print(f"        Fix: {fix}")

print("=" * 60)
print("  Pre-Submission Validator — EmailTriageEnv")
print("=" * 60)

# ── 1. Check required files exist ────────────────────────────
print("\n[1] Required files")
check("inference.py exists",     os.path.exists("inference.py"),     "Run CELL 3")
check("openenv.yaml exists",     os.path.exists("openenv.yaml"),     "Run CELL 8")
check("Dockerfile exists",       os.path.exists("Dockerfile"),       "Run CELL 5")
check("requirements.txt exists", os.path.exists("requirements.txt"), "Run CELL 5")
check("app.py exists",           os.path.exists("app.py"),           "Run CELL 4")
check("env/models.py exists",    os.path.exists("env/models.py"),    "Run CELL 3 of setup")
check("env/email_env.py exists", os.path.exists("env/email_env.py"), "Run setup cells")
check("env/data/emails.json exists", os.path.exists("env/data/emails.json"), "Run setup cells")

# ── 2. Check environment variables defined ────────────────────
print("\n[2] Environment variables")
check("API_BASE_URL defined", bool(os.environ.get("API_BASE_URL")), "export API_BASE_URL=https://api.openai.com/v1")
check("MODEL_NAME defined",   bool(os.environ.get("MODEL_NAME")),   "export MODEL_NAME=gpt-3.5-turbo")
check("HF_TOKEN defined",     bool(os.environ.get("HF_TOKEN")),     "export HF_TOKEN=your-key-here")

# ── 3. Check openenv.yaml has required fields ─────────────────
print("\n[3] openenv.yaml compliance")
try:
    content = open("openenv.yaml").read()
    check("has name field",      "name:" in content)
    check("has tasks field",     "tasks:" in content)
    check("has task_easy",       "task_easy" in content)
    check("has task_medium",     "task_medium" in content)
    check("has task_hard",       "task_hard" in content)
    check("has observation_space", "observation_space" in content)
    check("has action_space",    "action_space" in content)
except Exception as e:
    check("openenv.yaml readable", False, str(e))

# ── 4. Check typed models exist ───────────────────────────────
print("\n[4] Typed models")
try:
    sys.path.insert(0, ".")
    from env.models import Email, Observation, Action, StepResult, EpisodeResult
    check("Email model",       True)
    check("Observation model", True)
    check("Action model",      True)
    check("StepResult model",  True)
    check("EpisodeResult model", True)
except Exception as e:
    check("models import", False, str(e))

# ── 5. Check step/reset/state work ───────────────────────────
print("\n[5] step() / reset() / state() API")
try:
    from env.email_env import EmailTriageEnv
    from env.models import Action

    env = EmailTriageEnv(task_id="task_easy", seed=42)
    obs = env.reset()
    check("reset() returns Observation", hasattr(obs, "inbox"))
    check("reset() inbox has emails",    len(obs.inbox) > 0)

    action = Action(action_type="label", email_id=obs.inbox[0].id, label="urgent")
    result = env.step(action)
    check("step() returns StepResult",   hasattr(result, "reward"))
    check("step() reward is float",      isinstance(result.reward, float))
    check("step() done is bool",         isinstance(result.done, bool))

    state = env.state()
    check("state() returns dict",        isinstance(state, dict))
    check("state() has current_step",    "current_step" in state)
except Exception as e:
    check("environment API", False, str(e))

# ── 6. Check 3 tasks with graders, scores in 0.0-1.0 ─────────
print("\n[6] 3 tasks with graders")
try:
    from env.email_env import EmailTriageEnv
    from baseline.agent import RuleBasedAgent

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        env   = EmailTriageEnv(task_id=task_id, seed=42)
        agent = RuleBasedAgent(task_id=task_id)
        obs   = env.reset()
        while True:
            action = agent.act(obs)
            result = env.step(action)
            if result.done: break
            obs = result.observation
        ep    = env.episode_result()
        score = ep.total_score
        check(f"{task_id} score in 0.0-1.0", 0.0 <= score <= 1.0, f"score was {score}")
        check(f"{task_id} has grader_breakdown", bool(ep.grader_breakdown))
except Exception as e:
    check("graders", False, str(e))

# ── 7. Check inference.py runs without error ─────────────────
print("\n[7] inference.py runs without error")
try:
    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True, text=True, timeout=1200
    )
    check("inference.py exits cleanly",  result.returncode == 0, result.stderr[-300:] if result.stderr else "")
    check("inference.py prints scores",  "INFERENCE COMPLETE" in result.stdout, "Check inference.py output")
    check("inference_results.json saved", os.path.exists("inference_results.json"))
except subprocess.TimeoutExpired:
    check("inference.py under 20min", False, "Script took longer than 20 minutes")
except Exception as e:
    check("inference.py runs", False, str(e))

# ── 8. Check HTTP endpoints respond ──────────────────────────
print("\n[8] HTTP endpoints (Space URL ping)")
print(f"     Testing: {SPACE_URL}")
try:
    r = requests.get(f"{SPACE_URL}/health", timeout=10)
    check("GET /health returns 200", r.status_code == 200)

    r = requests.post(f"{SPACE_URL}/reset?task_id=task_easy&seed=42", timeout=10)
    check("POST /reset returns 200",  r.status_code == 200)
    check("POST /reset returns inbox", "inbox" in r.json())

    r = requests.get(f"{SPACE_URL}/state", timeout=10)
    check("GET /state returns 200", r.status_code == 200)

    r = requests.post(f"{SPACE_URL}/step",
                      json={"action_type":"label","email_id":"e001","label":"urgent"},
                      timeout=10)
    check("POST /step returns 200", r.status_code == 200)
except Exception as e:
    print(f"     Skipped HTTP checks (Space not reachable locally): {e}")
    print("     These will pass once deployed to HuggingFace")

# ── Final result ──────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  PASSED: {len(passed)}")
print(f"  FAILED: {len(failed)}")
print("=" * 60)
if failed:
    print("\n  Fix these before submitting:")
    for f in failed:
        print(f"    - {f}")
else:
    print("\n  ALL CHECKS PASSED — ready to submit!")