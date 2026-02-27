# NanoGUI

# Divide and Ground: A Lightweight Multi-Agent Framework for GUI Task Execution with Small Vision-Language Models

**Course:** Deep Learning with Language and Vision
**Date:** February 2026

---

## Motivation

Graphical User Interface (GUI) agents — systems that autonomously navigate and interact with software interfaces — have seen rapid progress driven by large proprietary models such as GPT-4V and Gemini. However, these models are computationally expensive and inaccessible for on-device or resource-constrained deployment. Recent work such as SeeClick [1] and OS-Atlas [2] has shown that small Vision-Language Models (VLMs, 2–7B parameters) can achieve competitive grounding performance when fine-tuned on GUI-specific data. Yet a core limitation remains: a single small VLM asked to simultaneously plan a multi-step task, localize UI elements, and verify its own actions is being asked to do too much at once, leading to compounding errors in long-horizon tasks.

This project addresses that gap by decomposing GUI task execution into a **three-agent pipeline** — Planner, Grounder, and Critic — each implemented with a small, resource-efficient VLM or lightweight classifier. We hypothesize that specialization across agents allows each module to excel at a narrower sub-problem, improving overall task success while remaining executable on a single consumer GPU (≤16 GB VRAM). 

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                    │
│        Natural Language Task Instruction + Screenshot           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │      PLANNER AGENT      │
              │       (VLM / LLM)       │
              │                         │
              │  Input:  Screenshot +   │
              │          Task String    │
              │  Output: Ordered list   │
              │          of subtasks    │
              └────────────┬────────────┘
                           │  Subtask_i (text)
                           ▼
              ┌─────────────────────────┐
              │      GROUNDER AGENT     │
              │     (LoRA-tuned VLM)    │
              │                         │
              │  Input:  Screenshot +   │
              │          Subtask_i      │
              │  Output: (x, y) click   │
              │          coordinate or  │
              │          bounding box   │
              └────────────┬────────────┘
                           │  Proposed Action
                           ▼
              ┌─────────────────────────┐
              │       CRITIC AGENT      │
              │  (Lightweight binary    │
              │   classifier / VLM)     │
              │                         │
              │  Input:  Screenshot +   │
              │          Action overlay │
              │  Output: Accept / Reject│
              └────────────┬────────────┘
                           │
               ┌───────────┴───────────┐
               │ Accept                │ Reject
               ▼                       ▼
     ┌──────────────────┐    ┌──────────────────────┐
     │  ACTION EXECUTOR │    │  Re-query Grounder   │
     │  Perform click / │    │  with Critic feedback│
     │  type / scroll   │    │  (up to N retries)   │
     └────────┬─────────┘    └──────────────────────┘
              │
              ▼
     Next screenshot → back to Planner
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                   │
│       Completed Task  /  Action Trajectory  /  Failure Report  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technical Approach

**Planner Agent:** A prompted small VLM receives the full screenshot and task instruction and produces a structured, ordered list of subtasks in natural language. No fine-tuning is required here; chain-of-thought prompting suffices for decomposition.

**Grounder Agent:** A Qwen2-VL-2B model fine-tuned via LoRA [3] on the ScreenSpot dataset [1], which contains 1200+ GUI grounding samples across iOS, Android, Web, and Desktop. The Grounder receives a single subtask string and a screenshot, and predicts the (x, y) coordinate of the UI element to interact with. LoRA rank r=16 keeps trainable parameters under 20M.

**Critic Agent:** A lightweight binary classifier that receives the screenshot with the proposed action rendered as a bounding-box overlay, and outputs a confidence score. Actions below a threshold trigger a retry with the Critic's rejection signal passed back to the Grounder as additional context.

---

## Evaluation Plan

| Metric | Description | Benchmark |
|---|---|---|
| **Grounding Accuracy** | % of subtasks where predicted click falls inside the correct element bounding box | ScreenSpot [1] |
| **Task Success Rate** | % of complete multi-step tasks finished correctly end-to-end | Mind2Web subset [4] |
| **Critic Precision/Recall** | How accurately the Critic distinguishes correct vs. incorrect actions | Held-out ScreenSpot split |
| **Efficiency** | GPU memory (GB), inference latency (ms/step), total trainable parameters | Measured locally |

**Ablation study (key experiment):** We compare four configurations to isolate each agent's contribution:
1. Grounder-only baseline (no Planner, no Critic)
2. Planner + Grounder (no Critic)
3. Grounder + Critic (no Planner)
4. Full pipeline: Planner + Grounder + Critic

---

## References

[1] Cheng, K., et al. *SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents.* ACL 2024. https://arxiv.org/abs/2401.10935

[2] Wu, Z., et al. *OS-Atlas: A Foundation Action Model for Generalist GUI Agents.* NeurIPS 2024. https://arxiv.org/abs/2410.23218

[3] Hu, E., et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. https://arxiv.org/abs/2106.09685

[4] Deng, X., et al. *Mind2Web: Towards a Generalist Agent for the Web.* NeurIPS 2023. https://arxiv.org/abs/2306.06070

[5] Lin, K.Q., et al. *ShowUI: One Vision-Language-Action Model for GUI Visual Agent.* arXiv 2024. https://arxiv.org/abs/2411.17465

[6] Wang, P., et al. *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution.* arXiv 2024. https://arxiv.org/abs/2409.12191

[7] Liu, H., et al. *Visual Instruction Tuning (LLaVA).* NeurIPS 2023. https://arxiv.org/abs/2304.08485
