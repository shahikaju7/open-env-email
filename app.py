import json
import gradio as gr
from env.email_env import EmailTriageEnv
from baseline.agent import RuleBasedAgent


def run_episode(task_id, seed):
    env   = EmailTriageEnv(task_id=task_id, seed=int(seed))
    agent = RuleBasedAgent(task_id=task_id)
    obs   = env.reset()
    log   = []

    while True:
        action = agent.act(obs)
        result = env.step(action)
        if action.action_type != "done":
            log.append({
                "step":        result.observation.current_step,
                "action":      action.action_type,
                "email_id":    action.email_id,
                "label":       action.label,
                "priority":    action.priority,
                "reward":      result.reward,
            })
        obs = result.observation
        if result.done:
            break

    episode   = env.episode_result()
    breakdown = episode.grader_breakdown
    breakdown["steps_taken"] = episode.steps_taken
    return json.dumps(breakdown, indent=2), json.dumps(log, indent=2)


def run_benchmark(seed):
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


with gr.Blocks(title="EmailTriageEnv") as demo:
    gr.Markdown("# EmailTriageEnv\nRule-based agent. No API key needed.")

    with gr.Tab("Run episode"):
        with gr.Row():
            task_in  = gr.Dropdown(choices=["task_easy","task_medium","task_hard"], value="task_easy", label="Task")
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

demo.launch(server_name="0.0.0.0", server_port=7860)