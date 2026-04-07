"""
NanoGUI Agents Module

This module provides the three core agents for GUI automation:
- PlannerAgent: Decomposes tasks into atomic sub-goals
- GrounderAgent: Maps sub-goals to grounded actions with coordinates
- VerifierAgent: Validates action success with two-stage verification

All agents inherit from BaseAgent for common functionality.

Usage:
------
```python
from NanoGUI.agents import PlannerAgent, GrounderAgent, VerifierAgent
from NanoGUI.models import create_glm_client

# Initialize model client
client = create_glm_client()

# Create agents
planner = PlannerAgent(model_client=client, max_steps=20)
grounder = GrounderAgent(model_client=client)
verifier = VerifierAgent(model_client=client, pixel_similarity_threshold=5.0)

# Use agents
sub_goals = await planner.plan(task, screenshot)
action = await grounder.ground(sub_goals[0], screenshot)
result = await verifier.verify(sub_goals[0], str(action), before_screenshot, after_screenshot)
```
"""

from .base import (
    # Base classes
    BaseAgent,

    # Data structures
    ActionResult,
    VerificationResult,
    GroundedAction,
    SubGoal,
    Coordinate,

    # Enums
    AgentType,
    ActionType,
    VerificationStatus,

    # Configuration
    AgentConfig,
    PlannerConfig,
    GrounderConfig,
    VerifierConfig,
    create_agent_config,

    # Exceptions
    AgentError,
    ParsingError,
    ExecutionError,
    VerificationError,

    # Utilities
    log_agent_call,
)

from .Planner import PlannerAgent
from .Grounder import GrounderAgent
from .Verifier import VerifierAgent

__all__ = [
    # Agents
    "PlannerAgent",
    "GrounderAgent",
    "VerifierAgent",
    "BaseAgent",

    # Data structures
    "ActionResult",
    "VerificationResult",
    "GroundedAction",
    "SubGoal",
    "Coordinate",

    # Enums
    "AgentType",
    "ActionType",
    "VerificationStatus",

    # Configuration
    "AgentConfig",
    "PlannerConfig",
    "GrounderConfig",
    "VerifierConfig",
    "create_agent_config",

    # Exceptions
    "AgentError",
    "ParsingError",
    "ExecutionError",
    "VerificationError",

    # Utilities
    "log_agent_call",
]

__version__ = "0.1.0"
