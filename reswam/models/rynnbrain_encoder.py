"""RynnBrain-1.1-2B 24-vision / 16-text VLM context encoder for ResWAM."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

RYNN_REPO = "Alibaba-DAMO-Academy/RynnBrain1.1-2B"
RYNN_REVISION = "36069aa841fc58f500d307316cbfdf379777ce8a"
TEXT_LAYERS = 16
PUBLISHED_TEXT_LAYERS = 24
VISION_LAYERS = 24
TEXT_HIDDEN_SIZE = 2048
VISION_HIDDEN_SIZE = 1024
EXPECTED_LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
    for index in range(PUBLISHED_TEXT_LAYERS)
)
LEGACY_ACTION_PLACEHOLDER = "<ACT_PH>"

DEFAULT_LIBERO_TEMPORAL_COT_PROMPT = (
    "Your task is {instruction}.\n\n"
    "From main-view observations, infer the robot arm's motion."
)
DEFAULT_LIBERO_POLICY_COT_PROMPT = (
    "Then use observations and the inferred motion to produce the robot policy actions."
)


def assert_no_legacy_action_placeholder(input_ids: torch.Tensor, processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", processor)
    token_id = tokenizer.convert_tokens_to_ids(LEGACY_ACTION_PLACEHOLDER)
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is not None and token_id != unknown_id:
        if bool(input_ids.eq(int(token_id)).any()):
            raise RuntimeError("ResWAM input_ids contain forbidden <ACT_PH>")


def build_libero_cot_messages(
    *,
    primary_images: Sequence[Sequence[Any]],
    wrist_images: Sequence[Sequence[Any]],
    instructions: Sequence[str],
) -> list[list[dict[str, Any]]]:
    if not (len(primary_images) == len(wrist_images) == len(instructions)):
        raise ValueError("LIBERO observation batch fields must have equal length")
    messages = []
    for imgs, wrist_imgs, instruction in zip(primary_images, wrist_images, instructions):
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in imgs]
        text_before = DEFAULT_LIBERO_TEMPORAL_COT_PROMPT.replace("{instruction}", str(instruction))
        content.append({"type": "text", "text": text_before})
        for w_img in wrist_imgs:
            content.append({"type": "image", "image": w_img})
        content.append({"type": "text", "text": DEFAULT_LIBERO_POLICY_COT_PROMPT})
        messages.append([{"role": "user", "content": content}])
    return messages


def truncated_rynnbrain_config(checkpoint: str | Path, revision: str | None = None):
    """Build a native Qwen3.5 config whose text layer indices are truncated to 0..15."""
    from transformers import AutoConfig

    is_local = Path(str(checkpoint)).exists()
    config = AutoConfig.from_pretrained(
        str(checkpoint),
        revision=None if is_local else revision,
        local_files_only=is_local,
        trust_remote_code=True,
    )
    text = config.text_config
    if (
        text.num_hidden_layers != PUBLISHED_TEXT_LAYERS
        or tuple(text.layer_types) != EXPECTED_LAYER_TYPES
    ):
        raise RuntimeError("Published RynnBrain text topology does not match expected 24-layer hybrid spec")
    text.num_hidden_layers = TEXT_LAYERS
    text.layer_types = list(EXPECTED_LAYER_TYPES[:TEXT_LAYERS])
    config.text_config = text
    config.use_cache = False
    text.use_cache = False
    return config


class RynnBrainText16Encoder(nn.Module):
    """Native RynnBrain-1.1-2B encoder with 24 vision blocks and 16 text layers.

    Supports both multi-camera prompt formatting ("{view} camera:" + image per view, then instruction)
    and LIBERO CoT prompt formatting.
    """

    def __init__(
        self,
        checkpoint: str | Path = RYNN_REPO,
        *,
        revision: str | None = RYNN_REVISION,
        views: Sequence[str] = ("head", "left_wrist", "right_wrist"),
        prompt_style: str = "multi_camera",
        dtype: torch.dtype = torch.bfloat16,
        activation_checkpointing: bool = False,
        meta_init: bool = False,
        vlm: nn.Module | None = None,
        processor: Any | None = None,
    ) -> None:
        super().__init__()
        self.views = tuple(views)
        self.prompt_style = str(prompt_style)
        self.dtype = dtype

        if vlm is not None and processor is not None:
            self.processor = processor
            self.vlm = vlm
            return

        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        is_local = Path(str(checkpoint)).exists()
        self.processor = AutoProcessor.from_pretrained(
            str(checkpoint),
            revision=None if is_local else revision,
            local_files_only=is_local,
            trust_remote_code=True,
        )
        config = truncated_rynnbrain_config(checkpoint, revision=revision)
        if meta_init:
            with torch.device("meta"):
                self.vlm = Qwen3_5ForConditionalGeneration(config)
        else:
            self.vlm = Qwen3_5ForConditionalGeneration.from_pretrained(
                str(checkpoint),
                revision=None if is_local else revision,
                config=config,
                dtype=dtype,
                attn_implementation="sdpa",
                device_map="cpu",
                low_cpu_mem_usage=True,
                local_files_only=is_local,
                trust_remote_code=True,
            )
        layers = self.vlm.model.language_model.layers
        vision_blocks = self.vlm.model.visual.blocks
        if len(layers) != TEXT_LAYERS or len(vision_blocks) != VISION_LAYERS:
            raise RuntimeError(
                f"RynnBrain topology mismatch: got {len(layers)} text layers and {len(vision_blocks)} vision blocks"
            )
        self.vlm.lm_head = None
        self.vlm.requires_grad_(True)
        if activation_checkpointing:
            self.vlm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": False}
            )

    def prepare(
        self,
        cameras: Mapping[str, Sequence[np.ndarray]],
        instructions: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if tuple(cameras) != self.views:
            raise ValueError(f"RynnBrain camera order must be {self.views}, got {tuple(cameras)}")
        if self.prompt_style == "libero_cot" and len(self.views) >= 2:
            primary_view, wrist_view = self.views[0], self.views[1]
            primary = [
                [Image.fromarray(np.asarray(x, dtype=np.uint8), mode="RGB")]
                for x in cameras[primary_view]
            ]
            wrist = [
                [Image.fromarray(np.asarray(x, dtype=np.uint8), mode="RGB")]
                for x in cameras[wrist_view]
            ]
            messages = build_libero_cot_messages(
                primary_images=primary,
                wrist_images=wrist,
                instructions=instructions,
            )
        else:
            messages = []
            for index, instruction in enumerate(instructions):
                content: list[dict[str, Any]] = []
                for name in self.views:
                    content.extend(
                        [
                            {"type": "text", "text": f"{name} camera:"},
                            {
                                "type": "image",
                                "image": Image.fromarray(
                                    np.asarray(cameras[name][index], dtype=np.uint8), "RGB"
                                ),
                            },
                        ]
                    )
                content.append({"type": "text", "text": str(instruction)})
                messages.append([{"role": "user", "content": content}])
        prepared = {
            key: value
            for key, value in self.processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).items()
            if torch.is_tensor(value)
        }
        assert_no_legacy_action_placeholder(prepared["input_ids"], self.processor)
        return prepared

    def forward_prepared(
        self, prepared: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.vlm.parameters()).device
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in prepared.items()
            if value is not None
        }
        mask = batch["attention_mask"]
        padding = (-int(mask.shape[1])) % 4
        if padding:
            pad_id = getattr(self.processor.tokenizer, "pad_token_id", 0) or 0
            batch["input_ids"] = F.pad(batch["input_ids"], (0, padding), value=pad_id)
            mask = F.pad(mask, (0, padding), value=0)
        autocast = (
            torch.autocast("cuda", dtype=self.dtype)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            output = self.vlm.model(
                input_ids=batch["input_ids"],
                attention_mask=mask,
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch.get("image_grid_thw"),
                use_cache=False,
            )
        hidden = output.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[:2] != mask.shape or hidden.shape[-1] != TEXT_HIDDEN_SIZE:
            raise RuntimeError(f"Invalid RynnBrain text16 context shape {tuple(hidden.shape)}")
        return hidden, mask.bool()

    def forward(
        self, prepared: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if any(
            token in key for key in prepared for token in ("future", "endpoint", "target")
        ):
            raise RuntimeError("Future camera observations are forbidden from entering RynnBrain VLM")
        return self.forward_prepared(prepared)
