import os

import pytest
import torch

from flagscale.models.vlm.qwen3_vl import DEFAULT_IMAGE_TOKEN
from flagscale.train.utils.image_tools import to_pil_preserve


def _load_processor():
    pytest.importorskip("transformers")
    from transformers import AutoProcessor

    model_id = os.environ.get("QWEN3_VL_TEST_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
    try:
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Unable to load processor for {model_id}: {exc}")


def test_apply_chat_template_batched_images_match_per_sample_messages():
    processor = _load_processor()
    batch_size = 2
    num_images = 2
    height = 32
    width = 32
    images = torch.rand(batch_size, num_images, 3, height, width)
    pil_images = [
        [to_pil_preserve(img.permute(1, 2, 0).numpy()) for img in sample] for sample in images
    ]

    instruction = "Describe."
    per_sample_messages = []
    for sample_images in pil_images:
        content = [{"type": "image", "image": img} for img in sample_images]
        content.append({"type": "text", "text": instruction})
        per_sample_messages.append({"role": "user", "content": content})

    rendered_from_messages = processor.apply_chat_template(
        per_sample_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt = f"{DEFAULT_IMAGE_TOKEN}\n" * num_images + instruction
    batched_messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ] * batch_size
    rendered_from_prefix = processor.apply_chat_template(
        batched_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    list_inputs = processor(
        text=rendered_from_messages,
        images=[img for sample in pil_images for img in sample],
        padding=True,
        return_tensors="pt",
    )
    batched_inputs = processor(
        text=rendered_from_prefix,
        images=images.view(-1, 3, height, width),
        padding=True,
        return_tensors="pt",
    )
    assert torch.equal(list_inputs["input_ids"], batched_inputs["input_ids"])
