"""
Centralized configuration for NanoGUI.

Provides:
- NanoGUIConfig: Single dataclass holding all settings
- load_config: Load from a YAML file with fallback to defaults
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class NanoGUIConfig:
    """Top-level configuration for the entire NanoGUI system."""

    # -- Model settings --
    planner_model: str = "glm-4.6v"
    grounder_model: str = "Qwen/Qwen2.5-VL-2B-Instruct"
    verifier_model: str = "HuggingFaceTB/SmolVLM-256M-Instruct"

    # -- Device settings --
    device: str = "cuda"
    load_in_4bit: bool = True

    # -- Pipeline settings --
    max_steps: int = 20
    max_retries: int = 3
    step_delay: float = 0.5
    verifier_threshold: float = 5.0

    # -- Action executor settings --
    action_delay: float = 0.5
    dry_run: bool = True          # True = simulate, no real clicks
    use_real_actions: bool = False  # True = real GUI, False = simulation

    # -- Screen capture --
    monitor: int = 1

    # -- Logging --
    log_level: str = "INFO"


def load_config(path: str | Path | None = None) -> NanoGUIConfig:
    """
    Load configuration from a YAML file.

    If the file does not exist or is None, returns defaults.

    Args:
        path: Path to config.yaml. Defaults to project root.

    Returns:
        Populated NanoGUIConfig.
    """
    defaults = NanoGUIConfig()

    if path is None:
        # Default: look for config.yaml next to the NanoGUI package
        path = Path(__file__).parent.parent / "config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        logger.info("No config file at %s — using defaults", path)
        return defaults

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed — using defaults")
        return defaults

    with open(path, "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}

    # Apply overrides that match NanoGUIConfig fields
    valid_fields = {f.name for f in defaults.__dataclass_fields__.values()}
    for key, value in overrides.items():
        if key in valid_fields:
            setattr(defaults, key, value)
        else:
            logger.warning("Ignoring unknown config key: %s", key)

    logger.info("Loaded config from %s", path)
    return defaults
