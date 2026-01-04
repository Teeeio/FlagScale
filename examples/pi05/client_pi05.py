import argparse
import base64
import io
import json
import os
import random
import sys
import time

from pathlib import Path
from typing import Any, Dict

import numpy as np
import requests

from PIL import Image

# Pi0.5 specific image dimensions (224x224 for vision transformer)
IMG_WIDTH = 224
IMG_HEIGHT = 224


def encode_image(path: str) -> str:
    """Read image as base64 string."""
    path = Path(path)
    if not path.exists():
        print(f"[WARNING] Image not found: {path.resolve()}. Using fake image.")
        # Create fake image for Pi0.5 (224x224)
        image = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=75)
        buffer.seek(0)
        jpeg_binary = buffer.read()
        return base64.b64encode(jpeg_binary).decode("utf-8")
    else:
        # Load and resize image to Pi0.5 expected size
        image = Image.open(path).convert('RGB')
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=75)
        buffer.seek(0)
        jpeg_binary = buffer.read()
        return base64.b64encode(jpeg_binary).decode("utf-8")


def check_health(base_url: str) -> None:
    """Ping /health; raise RuntimeError if unhealthy."""
    try:
        print(f"[*] Checking Pi0.5 server health at {base_url}/health")
        r = requests.get(f"{base_url}/health", timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Health-check request failed: {e}") from e

    data = r.json()
    if not (data.get("status") == "healthy" and data.get("model_loaded")):
        raise RuntimeError(f"Pi0.5 server not ready: {json.dumps(data, indent=2)}")

    model_type = data.get('model_type', 'Unknown')
    gpu_info = data.get('gpu_info', {})
    device_name = gpu_info.get('device_name', 'Unknown')

    print(f"[√] Pi0.5 server healthy")
    print(f"    Model Type: {model_type}")
    print(f"    GPU: {device_name}")
    if 'memory_allocated' in gpu_info:
        print(f"    GPU Memory: {gpu_info['memory_allocated']:.2f}GB allocated")


def build_payload(args) -> Dict[str, Any]:
    """Construct JSON payload for Pi0.5 /infer."""
    batch_size = 1

    # 1. Robot states (8-dimensional for Pi0.5)
    states = np.random.uniform(-1, 1, size=(batch_size, args.state_dim)).astype(np.float32)

    # 2. Robot actions (action horizon x action dim)
    actions = np.random.uniform(
        -1, 1, size=(batch_size, args.action_horizon, args.action_dim)
    ).astype(np.float32)

    # 3. Encode images (multi-camera support)
    img_sample = {}

    # Use provided images or create fake ones
    if args.base_img:
        img_sample["cam_high"] = encode_image(args.base_img)
    else:
        img_sample["cam_high"] = encode_image("fake_base.jpg")

    if args.left_wrist_img:
        img_sample["cam_left_wrist"] = encode_image(args.left_wrist_img)
    else:
        img_sample["cam_left_wrist"] = encode_image("fake_left_wrist.jpg")

    if args.right_wrist_img:
        img_sample["cam_right_wrist"] = encode_image(args.right_wrist_img)
    else:
        img_sample["cam_right_wrist"] = encode_image("fake_right_wrist.jpg")

    # Pi0.5 specific parameters
    payload = {
        "instruction": args.instruction,
        "states": states.tolist(),
        "actions": actions.tolist(),
        "images": [img_sample],
        # Pi0.5 specific generation parameters
        "num_steps": args.num_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        # Additional Pi0.5 parameters
        "action_dim": args.action_dim,
        "action_horizon": args.action_horizon,
        "state_dim": args.state_dim,
        "flow_matching_noise": args.flow_matching_noise,
    }

    return payload


def pretty_print_resp(resp: requests.Response) -> None:
    """Nicely print JSON response."""
    try:
        data = resp.json()
        if data.get("success"):
            print("[√] Inference successful!")
            print(json.dumps(data, indent=2, ensure_ascii=False))

            # Additional info if predicted actions are available
            if "predicted_actions" in data:
                actions = data["predicted_actions"]
                if isinstance(actions, list) and len(actions) > 0:
                    print(
                        f"\n[INFO] Predicted Actions Shape: [{len(actions)}, {len(actions[0]) if isinstance(actions[0], list) else 'N/A'}]"
                    )
                    if isinstance(actions[0], list):
                        print(f"[INFO] First action: {actions[0]}")
        else:
            print("[ERROR] Inference failed!")
            print(json.dumps(data, indent=2, ensure_ascii=False))
    except ValueError:
        print(resp.text)


def main():
    parser = argparse.ArgumentParser(description="Client for Pi0.5 inference API")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host of Pi0.5 server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port of Pi0.5 server (default: 5000)"
    )

    # Image inputs (optional)
    parser.add_argument("--base-img", help="Path to base camera RGB image (optional)")
    parser.add_argument("--left-wrist-img", help="Path to left wrist camera RGB image (optional)")
    parser.add_argument("--right-wrist-img", help="Path to right wrist camera RGB image (optional)")

    # Pi0.5 specific parameters
    parser.add_argument(
        "--state-dim",
        type=int,
        default=8,
        help="Dimension of robot state vector (default: 8 for Pi0.5)",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=32,
        help="Dimension of robot action vector (default: 32 for Pi0.5)",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=10,
        help="Action prediction horizon (default: 10 for Pi0.5)",
    )

    # Generation parameters
    parser.add_argument(
        "--instruction",
        default="Pick up the red object and place it in the container.",
        help="Task instruction for Pi0.5",
    )
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")

    # Pi0.5 specific flow matching parameter
    parser.add_argument(
        "--flow-matching-noise",
        type=float,
        default=0.1,
        help="Noise level for flow matching (default: 0.1)",
    )

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"-> Pi0.5 endpoint: {base_url}")
    print(f"-> Task: {args.instruction}")
    print()

    # Check server health first
    try:
        check_health(base_url)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print()

    # Build and send inference request
    payload = build_payload(args)
    print(f"[*] Sending inference request...")
    print(f"    State shape: [{args.state_dim}]")
    print(f"    Action shape: [{args.action_horizon}, {args.action_dim}]")
    print(
        f"    Images provided: {bool(args.base_img or args.left_wrist_img or args.right_wrist_img)}"
    )
    print()

    try:
        t0 = time.time()
        resp = requests.post(
            f"{base_url}/infer",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=300,
        )
        elapsed = (time.time() - t0) * 1000
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] HTTP request failed: {e}")
        sys.exit(1)

    print(f"[√] Response OK ({resp.status_code}) - {elapsed:.1f}ms")
    pretty_print_resp(resp)


if __name__ == "__main__":
    main()
