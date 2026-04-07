# NanoGUI - Multi-Agent GUI Automation Framework

A lightweight, modular framework for GUI task automation using specialized Vision-Language Models.

## Overview

NanoGUI decomposes GUI automation into three specialized agents:

- **Planner**: Breaks down complex tasks into atomic sub-goals
- **Grounder**: Predicts click coordinates for UI elements
- **Verifier**: Validates if actions succeeded

## Quick Start

```python
import asyncio
from NanoGUI import NanoGUIPipeline, load_test_sample

async def main():
    # Load test data
    screenshot, instruction, _ = load_test_sample()

    # Create pipeline
    pipeline = NanoGUIPipeline()

    # Execute task
    result = await pipeline.execute_task(instruction, screenshot)

    print(f"Success: {result.success}")
    print(f"Steps: {result.completed_steps}/{result.total_steps}")

asyncio.run(main())
```

## Installation

```bash
cd NanoGUI
pip install -r ../requirements.txt
```

## Configuration

Set your GLM API key:
```bash
export GLM_API_KEY="your-api-key-here"
```

## Running Tests

```bash
# Test individual agents
python tests/test_planner.py
python tests/test_grounder.py
python tests/test_verifier.py

# Run all tests
python tests/run_all_tests.py

# Run full pipeline
python pipeline.py
```

## Project Structure

```
NanoGUI/
├── agents/           # Agent implementations
│   ├── base.py       # Base agent class
│   ├── Planner.py    # Task decomposition
│   ├── Grounder.py   # Visual grounding
│   └── Verifier.py   # Action verification
├── models/           # Model clients
│   ├── glm_client.py # GLM cloud API
│   └── qwen_client.py # Local Qwen model
├── data/             # Data loading utilities
│   ├── test_data_loader.py
│   └── download_screenspot.py
├── configs/          # System prompts
│   └── sys_prompts/
├── tests/            # Test scripts
└── pipeline.py       # Main orchestrator
```

## Phase 1 Features

✅ Three-agent pipeline (Planner → Grounder → Verifier)
✅ ScreenSpot dataset integration
✅ Simulated action execution
✅ Comprehensive logging
✅ Error handling and retry logic

## Next Steps (Phase 2)

🔄 LoRA fine-tuning for Grounder
🔄 Real GUI action execution
🔄 End-to-end evaluation on benchmarks

## License

MIT License - See LICENSE file for details
