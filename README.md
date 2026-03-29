---
title: EmailTriageEnv
emoji: 📬
colorFrom: purple
colorTo: teal
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

OpenEnv-compliant email triage benchmark.

## Required Space Secrets

Add these in Settings → Repository Secrets:

| Secret Name | Description | Example |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model to use | `gpt-3.5-turbo` |
| `HF_TOKEN` | Your API key | `sk-...` or `hf_...` |

## API Endpoints

- `GET  /health` — ping check
- `POST /reset`  — start new episode
- `GET  /state`  — current state
- `POST /step`   — apply action

## Run inference locally

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-3.5-turbo
export HF_TOKEN=sk-...
python inference.py

