"""
models/glm_client.py

Thin factory that creates an AutoGen-compatible OpenAIChatCompletionClient
pointed at the GLM-4.6V endpoint (z.ai OpenAI-compatible API).

Usage
-----
    from models.glm_client import create_glm_client
    client = create_glm_client()
"""

import os
import logging
from autogen_ext.models.openai import OpenAIChatCompletionClient

logger = logging.getLogger(__name__)


def create_glm_client(
    model: str = "glm-4.6v",
    base_url: str = "https://api.z.ai/api/coding/paas/v4",
    api_key: str | None = None,
) -> OpenAIChatCompletionClient:
    """
    Create an AutoGen OpenAIChatCompletionClient configured for GLM-4.6V.

    Parameters
    ----------
    model    : GLM model string, default "glm-4.6v"
    base_url : z.ai OpenAI-compatible endpoint
    api_key  : If None, reads from GLM_API_KEY environment variable

    Returns
    -------
    OpenAIChatCompletionClient ready to pass to AssistantAgent
    """
    resolved_key = api_key or os.getenv("GLM_API_KEY")
    if not resolved_key:
        raise EnvironmentError(
            "GLM API key not found. Set the GLM_API_KEY environment variable "
            "or pass api_key= explicitly."
        )

    client = OpenAIChatCompletionClient(
        model=model,
        api_key=resolved_key,
        base_url=base_url,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": False,
            "family": False,
            "structured_output": True,
        },
    )
    logger.info("GLM client created — model=%s  endpoint=%s", model, base_url)
    return client


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    import logging
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.ui import Console

    logging.basicConfig(level=logging.WARNING)

    async def main() -> None:
        client = create_glm_client()

        agent = AssistantAgent(
            name="glm_smoke_test",
            model_client=client,
            system_message="You are a helpful assistant. Be concise.",
        )

        print("=" * 60)
        print("GLM client smoke-test — text only")
        print("=" * 60)
        await Console(agent.run_stream(task="Say hello and state your model name."))
        await client.close()

    asyncio.run(main())