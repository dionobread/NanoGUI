"""
NanoGUI - A lightweight multi-agent framework for GUI task execution.

This package provides a simple, clean pipeline for GUI automation using
specialized Vision-Language Models (VLMs).

Main Components:
- Planner: Decomposes tasks into sub-goals
- Grounder: Predicts click coordinates for UI elements
- Verifier: Validates if actions succeeded

Usage:
    from NanoGUI.pipeline import NanoGUIPipeline
    from NanoGUI.data import load_test_sample

    pipeline = NanoGUIPipeline()
    screenshot, instruction, _ = load_test_sample()
    result = await pipeline.execute_task(instruction, screenshot)
"""

__version__ = "0.1.0"

from .pipeline import NanoGUIPipeline, PipelineConfig, PipelineResult
from .agents import PlannerAgent, GrounderAgent, VerifierAgent
from .data import load_test_sample, get_dataset_stats
from .models import create_glm_client

__all__ = [
    "NanoGUIPipeline",
    "PipelineConfig",
    "PipelineResult",
    "PlannerAgent",
    "GrounderAgent",
    "VerifierAgent",
    "load_test_sample",
    "get_dataset_stats",
    "create_glm_client",
]
