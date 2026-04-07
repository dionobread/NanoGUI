"""
models/qwen_client.py
─────────────────────
A custom AutoGen ChatCompletionClient that runs Qwen2.5-VL locally via
HuggingFace Transformers.  Drop-in replacement for the GLM cloud client —
any agent that accepts create_glm_client() also accepts QwenVLClient().

Design notes
────────────
- Implements the full autogen_core.models.ChatCompletionClient abstract
  interface so AutoGen agents work without modification.
- Supports both text-only and multimodal (text + PIL image) messages.
- Optional 4-bit quantization via bitsandbytes to fit on a single 24 GB GPU.
- create_stream() falls back to a non-streaming wrapper (yields one chunk)
  because HF generate() is not natively streaming in this setup.

Usage
-----
    from models.qwen_client import QwenVLClient

    client = QwenVLClient(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        device="cuda",
        load_in_4bit=True,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, Mapping, Sequence

from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from autogen_core import CancellationToken
from autogen_core.models import ModelFamily

logger = logging.getLogger(__name__)


class QwenVLClient(ChatCompletionClient):
    """
    Local Qwen2.5-VL client for AutoGen.

    Parameters
    ----------
    model_name  : HuggingFace model id, e.g. "Qwen/Qwen2.5-VL-3B-Instruct"
    device      : "cuda", "cpu", or "mps"
    load_in_4bit: Use bitsandbytes 4-bit quantization (saves ~50 % VRAM)
    max_new_tokens : Hard cap on generated tokens
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        torch_dtype: str = "bfloat16",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._load_in_4bit = load_in_4bit
        self._max_new_tokens = max_new_tokens
        self._torch_dtype_str = torch_dtype

        # Lazy-load — model is not loaded until first .create() call
        self._model = None
        self._processor = None

        # Usage accumulators
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    # ── Lazy model loader ─────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load model and processor on first use."""
        if self._model is not None:
            return

        logger.info("Loading %s (device=%s, 4bit=%s) …",
                    self._model_name, self._device, self._load_in_4bit)

        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        torch_dtype = getattr(torch, self._torch_dtype_str)

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": "auto" if self._device == "cuda" else self._device,
        }

        if self._load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._model_name, **load_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        logger.info("Model loaded successfully.")

    # ── Message conversion ────────────────────────────────────────────────────

    def _build_messages(self, messages: Sequence[LLMMessage]) -> list[dict]:
        """
        Convert AutoGen LLMMessage list → Qwen chat template format.

        AutoGen message types handled:
            SystemMessage  → role "system"
            UserMessage    → role "user"  (content: str or list with Image)
            AssistantMessage → role "assistant"
        """
        from autogen_core import Image as AGImage

        chat_messages: list[dict] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                chat_messages.append({"role": "system", "content": msg.content})

            elif isinstance(msg, UserMessage):
                # Content can be a plain string or a list mixing str and Image
                if isinstance(msg.content, str):
                    chat_messages.append({"role": "user", "content": msg.content})
                else:
                    # Multimodal: build Qwen content list
                    content_parts: list[dict] = []
                    for part in msg.content:
                        if isinstance(part, str):
                            content_parts.append({"type": "text", "text": part})
                        elif isinstance(part, AGImage):
                            # AGImage wraps a PIL image
                            content_parts.append({
                                "type": "image",
                                "image": part.image,   # PIL.Image
                            })
                        else:
                            # Fallback: stringify unknown parts
                            content_parts.append({"type": "text", "text": str(part)})
                    chat_messages.append({"role": "user", "content": content_parts})

            elif isinstance(msg, AssistantMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                chat_messages.append({"role": "assistant", "content": content})

        return chat_messages

    # ── Core create() ─────────────────────────────────────────────────────────

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = None,
        json_output: bool | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        """Run inference synchronously in a thread pool to keep async loop free."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._create_sync, messages
        )

    def _create_sync(self, messages: Sequence[LLMMessage]) -> CreateResult:
        import torch

        self._load_model()

        chat_messages = self._build_messages(messages)

        # Build text prompt via chat template
        text_prompt = self._processor.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )

        # Collect PIL images from messages (in order they appear)
        from autogen_core import Image as AGImage
        images = []
        for msg in messages:
            if isinstance(msg, UserMessage) and isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, AGImage):
                        images.append(part.image)

        # Tokenize
        if images:
            from qwen_vl_utils import process_vision_info
            # Use Qwen's vision utilities for proper image preprocessing
            image_inputs, video_inputs = process_vision_info(chat_messages)
            inputs = self._processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self._processor(
                text=[text_prompt],
                padding=True,
                return_tensors="pt",
            )

        # Move to device
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[:, input_len:]
        response_text = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()

        completion_tokens = new_tokens.shape[1]

        # Update accumulators
        self._total_prompt_tokens += input_len
        self._total_completion_tokens += completion_tokens

        return CreateResult(
            finish_reason="stop",
            content=response_text,
            usage=RequestUsage(
                prompt_tokens=input_len,
                completion_tokens=completion_tokens,
            ),
            cached=False,
        )

    # ── Streaming (fallback — yields single chunk) ────────────────────────────

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = None,
        json_output: bool | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        """
        Streaming interface — runs full generation then yields result as one
        chunk.  This satisfies AutoGen's streaming protocol without requiring
        HF token-level streaming.
        """
        result = await self.create(messages)
        # Yield text content, then final CreateResult
        if isinstance(result.content, str):
            yield result.content
        yield result

    # ── Required abstract properties / methods ────────────────────────────────

    @property
    def model_info(self) -> ModelInfo:
        return {
            "vision": True,
            "function_calling": False,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        }

    @property
    def capabilities(self) -> ModelInfo:          # alias required by some versions
        return self.model_info

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._total_prompt_tokens,
            completion_tokens=self._total_completion_tokens,
        )

    def total_usage(self) -> RequestUsage:
        return self.actual_usage()

    def count_tokens(self, messages: Sequence[LLMMessage], **kwargs: Any) -> int:
        # Approximate — 4 chars per token
        total = sum(
            len(m.content) // 4 if isinstance(m.content, str) else 100
            for m in messages
        )
        return total

    def remaining_tokens(self, messages: Sequence[LLMMessage], **kwargs: Any) -> int:
        return max(0, 32768 - self.count_tokens(messages))

    async def close(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            import torch
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            if self._device == "cuda":
                torch.cuda.empty_cache()
            logger.info("QwenVLClient released model from memory.")


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    import logging
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.ui import Console

    logging.basicConfig(level=logging.INFO)

    # ── CONFIG: switch between a real GPU run and a mock dry-run ──────────────
    DRY_RUN = True   # Set False when you have a GPU + the model downloaded

    async def main() -> None:
        if DRY_RUN:
            # ── Dry-run: test the client interface without loading the model ──
            print("=" * 60)
            print("QwenVLClient dry-run (DRY_RUN=True — no model loaded)")
            print("=" * 60)

            from autogen_core.models import UserMessage

            # Instantiate but do NOT trigger model loading
            client = QwenVLClient(
                model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                device="cpu",
                load_in_4bit=False,
            )
            print(f"model_info  : {client.model_info}")
            print(f"count_tokens: {client.count_tokens([UserMessage(content='hello world', source='user')])}")
            print(f"remaining   : {client.remaining_tokens([UserMessage(content='hello world', source='user')])}")
            print("Interface checks passed. Set DRY_RUN=False to run real inference.")
            await client.close()

        else:
            # ── Real inference run ────────────────────────────────────────────
            print("=" * 60)
            print("QwenVLClient — real inference (Qwen2.5-VL-3B-Instruct)")
            print("=" * 60)

            client = QwenVLClient(
                model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                device="cuda",
                load_in_4bit=True,
            )
            agent = AssistantAgent(
                name="qwen_local_agent",
                model_client=client,
                system_message="You are a helpful assistant. Be concise.",
            )
            await Console(agent.run_stream(task="What is 2 + 2? Answer in one sentence."))
            await client.close()

    asyncio.run(main())