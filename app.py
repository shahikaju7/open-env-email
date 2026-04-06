import os
import json
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from env.email_env import EmailTriageEnv
from env.models import Action

# Initialize FastAPI app globally
api = FastAPI()
_env_store = {}


@api.get("/health")
def health():
    return {"status": "ok", "message": "EmailTriageEnv is running"}


@api.post("/reset")
def reset(task_id: str = "task_easy", seed: int = 42):
    env = EmailTriageEnv(task_id=task_id, seed=seed)
    _env_store["current"] = env
    obs = env.reset()
    return JSONResponse(content=obs.model_dump())


@api.get("/state")
def state():
    if "current" not in _env_store:
        env = EmailTriageEnv(task_id="task_easy", seed=42)
        _env_store["current"] = env
        env.reset()
    return JSONResponse(content=_env_store["current"].state())


@api.post("/step")
def step(action: dict):
    if "current" not in _env_store:
        env = EmailTriageEnv(task_id="task_easy", seed=42)
        _env_store["current"] = env
        env.reset()
    try:
        act    = Action(**action)
        result = _env_store["current"].step(act)
        return JSONResponse(content=result.model_dump())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


def run_episode(task_id, seed):
    from baseline.agent import RuleBasedAgent
    env   = EmailTriageEnv(task_id=task_id, seed=int(seed))
    agent = RuleBasedAgent(task_id=task_id)
    obs   = env.reset()
    log   = []
    while True:
        action = agent.act(obs)
        result = env.step(action)
        if action.action_type != "done":
            log.append({
                "step":     result.observation.current_step,
                "action":   action.action_type,
                "email_id": action.email_id,
                "label":    action.label,
                "reward":   result.reward,
            })
        obs = result.observation
        if result.done:
            break
    ep = env.episode_result()
    return json.dumps(ep.grader_breakdown, indent=2), json.dumps(log, indent=2)


def run_benchmark(seed):
    from baseline.agent import RuleBasedAgent
    rows = []
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        env   = EmailTriageEnv(task_id=task_id, seed=int(seed))
        agent = RuleBasedAgent(task_id=task_id)
        obs   = env.reset()
        while True:
            action = agent.act(obs)
            result = env.step(action)
            if result.done:
                break
            obs = result.observation
        ep = env.episode_result()
        rows.append([task_id, f"{ep.total_score:.4f}", str(ep.grader_breakdown)])
    return rows


def build_ui():
    with gr.Blocks(title="EmailTriageEnv") as ui:
        gr.Markdown("# EmailTriageEnv\nOpenEnv benchmark. API: /reset /step /state /health")
        with gr.Tab("Run episode"):
            with gr.Row():
                task_in  = gr.Dropdown(
                    choices=["task_easy", "task_medium", "task_hard"],
                    value="task_easy", label="Task"
                )
                seed_in  = gr.Number(value=42, label="Seed", precision=0)
                run_btn  = gr.Button("Run", variant="primary")
            with gr.Row():
                score_out = gr.Textbox(label="Score breakdown", lines=10)
                log_out   = gr.Textbox(label="Actions log",     lines=10)
            run_btn.click(run_episode, inputs=[task_in, seed_in], outputs=[score_out, log_out])
        with gr.Tab("Benchmark"):
            bench_seed = gr.Number(value=42, label="Seed", precision=0)
            bench_btn  = gr.Button("Run all 3 tasks", variant="primary")
            bench_out  = gr.Dataframe(headers=["Task", "Score", "Breakdown"])
            bench_btn.click(run_benchmark, inputs=[bench_seed], outputs=[bench_out])
    return ui


# Build the Gradio UI and mount it onto the FastAPI app globally
gradio_ui = build_ui()
app = gr.mount_gradio_app(api, gradio_ui, path="/")


# Removed: if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=7860)
