"""
Collator for Qwen2-VL outputs
"""


from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class GrounderCollator:
    """
    Collates multimodal batches for Qwen2-VL fine-tuning.

    Pads token sequences to the longest in the batch.
    Labels for prompt tokens are masked to -100 so loss is only
    computed on the assistant response.
    """
    processor: any  # Qwen2VLProcessor
    pad_token_id: int = 0

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = [item["input_ids"].squeeze(0) for item in batch]
        attention_masks = [item["attention_mask"].squeeze(0) for item in batch]

        # Pad to longest sequence in batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        # Mask prompt tokens from loss — only train on assistant response
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        }