readme = """---
title: EmailTriageEnv
emoji: 📬
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - agent
  - benchmark
  - reinforcement-learning
---

# EmailTriageEnv

A real-world OpenEnv environment where an AI agent triages a business inbox
by labeling, prioritizing, drafting replies, and escalating emails.

## Environment Description

The agent receives a business inbox of 10 emails and must process each one
correctly. Rewards are given per action based on accuracy.

## Observation Space

| Field | Type | Description |
|---|---|---|
| inbox | list | List of emails with id, subject, sender, body |
| current_step | int | Current step number |
| max_steps | int | Maximum steps allowed |
| task_id | string | Current task name |
| instructions | string | What the agent must do |
| score_so_far | float | Running score |

## Action Space

| Field | Type | Description |
|---|---|---|
| action_type | string | label / prioritize / reply / escalate / archive / done |
| email_id | string | ID of email to act on |
| label | string | urgent / billing / support / spam / hr / legal |
| priority | int | 1 (highest) to 5 (lowest) |
| reply_body | string | Reply text for support/billing emails |
| escalate_to | string | cto / legal-team / ceo |

## Tasks

| Task | Difficulty | Description | Max Steps |
|---|---|---|---|
| task_easy | Easy | Label 10 emails correctly | 15 |
| task_medium | Medium | Label + prioritize 10 emails | 25 |
| task_hard | Hard | Label + prioritize + reply + escalate | 40 |

## Reward Function

| Action | Reward |
|---|---|
| Correct label | +0.25 to +1.0 |
| Wrong label | -0.05 |
| Priority accuracy | up to +0.50 |
| Quality reply | up to +0.35 |
| Correct escalation | up to +0.20 |

## Setup Instructions
```bash
git clone https://huggingface.co/spaces/shahikajal7/email-triage-env
cd email-triage-env
pip install -r requirements.txt
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your-hf-token
python inference.py
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Check Space is alive |
| /reset | POST | Start new episode |
| /state | GET | Get current state |
| /step | POST | Apply one action |

## Required Space Secrets

| Name | Value |
|---|---|
| API_BASE_URL | https://router.huggingface.co/v1 |
| MODEL_NAME | meta-llama/Llama-3.1-8B-Instruct |
| HF_TOKEN | your hf_ token |
"""