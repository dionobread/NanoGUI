"""
Base classes and shared utilities for all NanoGUI agents.

This module provides:
- BaseAgent: Common functionality for Planner, Grounder, Verifier
- Shared data structures (ActionResult, AgentResponse)
- Common parsing utilities (JSON extraction, error handling)
- Configuration base classes
"""

import re
import abc
import json
import logging
from enum import Enum
from typing import Any
from dataclasses import dataclass, field

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

logger = logging.getLogger(__name__)


# Data Structures

class AgentType(Enum):
    """Enum for agent type identification."""
    PLANNER = "planner"
    GROUNDER = "grounded"
    VERIFIER = "verifier"


class ActionType(Enum):
    """Standardized action types for GUI automation."""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    KEY = "key"
    WAIT = "wait"
    DRAG = "drag"


class VerificationStatus(Enum):
    """Verification result status."""
    SUCCESS = "success"
    FAILURE = "failure"
    UNCERTAIN = "uncertain"


@dataclass
class Coordinate:
    """
    Normalized screen coordinate.

    x : float - Horizontal position [0.0, 1.0]
    y : float - Vertical position [0.0, 1.0]
    """
    x: float
    y: float

    def __post_init__(self):
        """Validate coordinate range."""
        if not (0.0 <= self.x <= 1.0):
            raise ValueError(f"x coordinate must be in [0, 1], got {self.x}")
        if not (0.0 <= self.y <= 1.0):
            raise ValueError(f"y coordinate must be in [0, 1], got {self.y}")

    def to_pixels(
        self,
        screen_width: int,
        screen_height: int
    ) -> tuple[int, int]:
        """Convert to absolute pixel coordinates."""
        return (
            int(self.x * screen_width),
            int(self.y * screen_height)
        )

    @classmethod
    def from_pixels(
        cls,
        x: int,
        y: int,
        screen_width: int,
        screen_height: int
    ) -> "Coordinate":
        """Create from absolute pixel coordinates."""
        return cls(
            x=x / screen_width,
            y=y / screen_height
        )

    def __repr__(self) -> str:
        return f"Coordinate({self.x:.3f}, {self.y:.3f})"


@dataclass
class ActionResult:
    """
    Standardized result from agent execution.

    All agents return this structure for consistency.
    """
    success: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"ActionResult({status} content={self.content[:50]}...)"


@dataclass
class VerificationResult:
    """
    Result from Verifier agent.

    status : VerificationStatus - Success, failure, or uncertain
    reason : str - Human-readable explanation
    confidence : float | None - Optional confidence score [0, 1]
    """
    status: VerificationStatus
    reason: str
    confidence: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == VerificationStatus.SUCCESS

    @property
    def failed(self) -> bool:
        return self.status == VerificationStatus.FAILURE

    @property
    def uncertain(self) -> bool:
        return self.status == VerificationStatus.UNCERTAIN

    def __repr__(self) -> str:
        conf_str = f" (conf={self.confidence:.2f})" if self.confidence else ""
        return f"VerificationResult(status={self.status.value}{conf_str}, reason={self.reason!r})"


@dataclass
class GroundedAction:
    """
    Structured output from the Grounder agent.

    action_type : ActionType - Type of action to execute
    coordinate : Coordinate | None - Normalized (x, y) for click/scroll
    text : str | None - Text to type or key to press
    direction : str | None - Scroll direction ("up" | "down")
    reasoning : str - Model's explanation for debugging
    """
    action_type: ActionType
    coordinate: Coordinate | None = None
    text: str | None = None
    direction: str | None = None
    reasoning: str = ""

    def to_pixel_coords(
        self,
        screen_width: int,
        screen_height: int
    ) -> tuple[int, int] | None:
        """Convert normalized coordinate to absolute pixel position."""
        if self.coordinate is None:
            return None
        return self.coordinate.to_pixels(screen_width, screen_height)

    def __repr__(self) -> str:
        if self.action_type == ActionType.CLICK:
            return f"GroundedAction(click @ {self.coordinate} | {self.reasoning})"
        if self.action_type == ActionType.TYPE:
            return f"GroundedAction(type '{self.text}')"
        if self.action_type == ActionType.KEY:
            return f"GroundedAction(key '{self.text}')"
        if self.action_type == ActionType.SCROLL:
            return f"GroundedAction(scroll {self.direction})"
        if self.action_type == ActionType.WAIT:
            return f"GroundedAction(wait)"
        return f"GroundedAction({self.action_type})"


@dataclass
class SubGoal:
    """
    A single atomic sub-goal from the Planner.

    description : str - Natural language description
    step_number : int | None - Optional ordering
    completed : bool - Whether this step has been executed
    """
    description: str
    step_number: int | None = None
    completed: bool = False

    def __repr__(self) -> str:
        status = "✓" if self.completed else "○"
        num = f"{self.step_number}. " if self.step_number else ""
        return f"SubGoal({status} {num}{self.description})"


# Base Agent Class

class BaseAgent(abc.ABC):
    """
    Abstract base class for all NanoGUI agents.

    Provides common functionality:
    - AutoGen AssistantAgent initialization
    - System prompt management
    - JSON output parsing
    - Logging and error handling
    - Cancellation token support

    Subclasses must implement:
    - _get_system_prompt() : Return the system prompt
    - async execute() : Main agent method
    """

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        system_message: str | None = None,
        reflect_on_tool_use: bool = False,
    ):
        """
        Initialize the base agent.

        name : str - Agent identifier (used for logging)
        model_client : ChatCompletionClient - AutoGen-compatible model client
        system_message : str | None - Override default system prompt
        reflect_on_tool_use : bool - Whether agent reflects on tool use
        """
        self._name = name
        self._model_client = model_client

        # Use provided system message or class default
        sys_msg = system_message or self.load_system_prompt()

        # Initialize AutoGen AssistantAgent
        self._agent = AssistantAgent(
            name=name,
            model_client=model_client,
            system_message=sys_msg,
            reflect_on_tool_use=reflect_on_tool_use,
        )

        logger.info("[%s] Agent initialized with model client", self._name)

    @abc.abstractmethod
    async def execute(self, *args, **kwargs) -> ActionResult:
        """
        Main execution method for the agent.

        Subclasses implement their specific logic here.
        """
        pass

    def load_system_prompt(self, prompt_name: str = "default") -> str:
        """
        Load system prompt from YAML configuration file.

        Args:
            agent_name: Name of the agent (planner, grounder, verifier)
            prompt_name: Name of the prompt to load (default, replan, etc.)

        Returns:
            The loaded system prompt as a string

        Raises:
            FileNotFoundError: If the config file doesn't exist
            KeyError: If the prompt_name is not found in the config
        """
        import os
        from pathlib import Path

        # Get the config directory relative to this file
        current_dir = Path(__file__).parent
        config_file = current_dir.parent / "configs" / "sys_prompts" / f"{self._name}.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Prompt config file not found: {config_file}")

        try:
            import yaml
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Determine which prompt to return
            if prompt_name == "default":
                if self._name == "planner":
                    return config.get("planner_prompt", "")
                elif self._name == "grounder":
                    return config.get("grounder_prompt", "")
                elif self._name == "verifier":
                    return config.get("verifier_prompt", "")
                else:
                    # Fallback to any prompt that matches
                    for key in config:
                        if key.endswith("_prompt"):
                            return config[key]
            else:
                # Look for specific prompt by name
                prompt_key = f"{prompt_name}_prompt"
                return config.get(prompt_key, "")

            return ""

        except ImportError:
            logger.warning(f"PyYAML not installed, returning empty prompt. Install with: pip install pyyaml")
            return ""
        except Exception as e:
            logger.error(f"Error loading prompt from {config_file}: {e}")
            return ""

    @staticmethod
    def extract_last_text(run_result) -> str:
        """
        Extract the last text message from an AutoGen TaskResult.

        This is a common pattern across all agents - pull the final
        assistant response from the message history.
        """
        for msg in reversed(run_result.messages):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                return msg.content
        return ""

    @staticmethod
    def parse_json_output(raw: str) -> dict[str, Any]:
        """
        Parse JSON output from model response.

        Handles common issues:
        - Markdown code fences (```json ... ```)
        - Leading/trailing whitespace
        - Malformed JSON (graceful fallback)

        Parameters
        ----------
        raw : str - Raw model output

        Returns
        -------
        dict - Parsed JSON data

        Raises
        ------
        ValueError - If JSON is invalid and no fallback available
        """
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        try:
            data = json.loads(cleaned)
            return data
        except json.JSONDecodeError as exc:
            logger.warning(
                "[%s] JSON parse failed: %s\nRaw output: %s",
                exc, raw[:200]
            )
            raise ValueError(
                f"Failed to parse model output as JSON. "
                f"Error: {exc}. Raw: {raw[:200]}"
            ) from exc

    @staticmethod
    def extract_coordinate_from_text(
        text: str,
        as_normalized: bool = True
    ) -> tuple[float, float] | None:
        """
        Extract (x, y) coordinate from natural language text.

        Handles formats:
        - "(450, 230)" - pixel coordinates
        - "450, 230" - pixel coordinates
        - "(0.45, 0.23)" - normalized coordinates
        - "click at 0.45, 0.23"

        Parameters: 
        text : str - Text to search
        as_normalized : bool - If True, expect normalized [0,1],
                              if False, expect pixels

        Returns:
        tuple[float, float] | None - (x, y) or None if not found
        """
        # Try normalized floats: (0.45, 0.23)
        if as_normalized:
            match = re.search(r"\(?(0\.\d+)[,\s]+(0\.\d+)\)?", text)
            if match:
                return float(match.group(1)), float(match.group(2))

        # Try integers: (450, 230) or 450, 230
        match = re.search(r"\(?(\d+)[,\s]+(\d+)\)?", text)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            if as_normalized:
                # Assume coordinates are pixels, need image size to normalize
                # Return as-is and let caller handle normalization
                return float(x), float(y)
            return x, y

        return None

    async def close(self) -> None:
        """
        Clean up resources.

        Override in subclasses if needed.
        """
        if hasattr(self._model_client, "close"):
            await self._model_client.close()
        logger.info("[%s] Agent closed", self._name)


# Configuration Base Classes

@dataclass
class AgentConfig:
    """Base configuration for all agents."""

    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str = "cuda"
    load_in_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.0

    # Validation
    def __post_init__(self):
        if self.device not in ("cuda", "cpu", "mps"):
            raise ValueError(f"Invalid device: {self.device}")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"Temperature must be in [0, 2], got {self.temperature}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")


@dataclass
class PlannerConfig(AgentConfig):
    """Configuration specific to Planner agent."""
    max_sub_goals: int = 20
    min_sub_goals: int = 1


@dataclass
class GrounderConfig(AgentConfig):
    """Configuration specific to Grounder agent."""
    # Use smaller model for performance
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    merge_weights: bool = True
    lora_adapter_path: str | None = "./grounder-lora-adapter"
    # added some extra things to configure during training
    load_in_4bit: bool = False
    lora_r: int = 16  # The paper says 16 for r
    lora_alpha: int = 32
    lora_target_modules: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]  # TODO: Double check


@dataclass
class VerifierConfig(AgentConfig):
    """Configuration specific to Verifier agent."""
    # Use tiny model for fast verification
    model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    pixel_similarity_threshold: float = 5.0
    confidence_threshold: float = 0.5


# Exceptions

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class ParsingError(AgentError):
    """Raised when agent output cannot be parsed."""
    pass


class ExecutionError(AgentError):
    """Raised when agent execution fails."""
    pass


class VerificationError(AgentError):
    """Raised when verification fails unexpectedly."""
    pass


# Utilities

def create_agent_config(
    agent_type: AgentType,
    **kwargs
) -> AgentConfig:
    """
    Factory function to create agent-specific configs.

    Parameters: 
    agent_type : AgentType - Type of agent
    **kwargs - Additional config overrides

    Returns: 
    AgentConfig subclass instance
    """
    config_classes = {
        AgentType.PLANNER: PlannerConfig,
        AgentType.GROUNDER: GrounderConfig,
        AgentType.VERIFIER: VerifierConfig,
    }

    config_class = config_classes.get(agent_type, AgentConfig)
    return config_class(**kwargs)


def log_agent_call(
    agent_name: str,
    action: str,
    **details
) -> None:
    """
    Standardized logging for agent calls.

    Parameters: 
    agent_name : str - Name of the agent
    action : str - Action being performed
    **details - Additional context to log
    """
    detail_str = " ".join(f"{k}={v}" for k, v in details.items())
    logger.info("[%s] %s %s", agent_name, action, detail_str)


# Exports

__all__ = [
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

    # Base class
    "BaseAgent",

    # Configs
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
