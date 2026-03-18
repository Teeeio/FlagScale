## Introduce

We introduce support for deploying large models with FlagScale, leveraging the Ray framework for efficient orchestration and scalability. Currently, this implementation supports the Qwen model, enabling users to easily deploy and manage large-scale machine learning services.

Future Key features include:

- Easy distributed Serve on base of eamless integration with Ray.
- Optimized resource management for large model inference.
- Simplified deployment process for the LLM and Multimodal models.

This enhancement will significantly improve the usability of FlagScale for large model deployment scenarios.

## Setup

[Install vLLM](../../README.md#setup)

## Prepare Model

[Prepare Qwen data](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/summary)

```shell
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /models/
```

## Serve run

```shell
cd FlagScale
flagscale serve qwen --config ./examples/qwen/conf/config_qwen2.5_7b.yaml
# or
flagscale serve qwen -c ./examples/qwen/conf/config_qwen2.5_7b.yaml
```

## Serve call

```shell
curl http://127.0.0.1:4567/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "/models/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Introduce Bruce Lee in details."}
        ]
    }'
```

## Serve stop

```shell
cd FlagScale
flagscale serve qwen --config ./examples/qwen/conf/config_qwen2.5_7b.yaml --stop
# or
flagscale serve qwen -c ./examples/qwen/conf/config_qwen2.5_7b.yaml --stop
```

## logs

Since serve is the distributed mode, the logs are stored separately. \
The default logs of are located in `/outputs`.\


## Config Template

Flagscale.serve will support multiple scenarios. For better performance and usage, Flagscale.serve will optimize for specific scenarios, and these optimizations can be applied through different configurations.

### Command Line Mode with vLLM

If origin model is executed in command line mode with vLLM, we can use Flagscale.serve to deploy it easily.

```shell
vllm serve /models/Qwen2.5-7B-Instruct --tensor-parallel-size=1 --gpu-memory-utilization=0.9 --max-model-len=32768 --max-num-seqs=256 --port=4567 --trust-remote-code --enable-chunked-prefill
```

All the args remain the same as vLLM.

```YAML
- serve_id: vllm_model
  engine: vllm
  engine_args:
    model: /models/Qwen2.5-7B-Instruct
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    max_model_len: 32768
    max_num_seqs: 256
    port: 4567
    trust_remote_code: true
    enable_chunked_prefill: true
```

### How to config serve parameters
***deploy*** block is used to specify the parameters of serve. The ***models*** block is used to specify the parameters of each model decorated by "serve.remote".

## Qwen-GR00T VLA Serving

The unified VLA serving entrypoint is `flagscale/serve/run_serve_vla.py`. It currently supports `QwenGr00t` only. `flagscale/serve/run_serve_qwen_gr00t.py` is removed; `qwen_gr00t` serving now goes through the unified entrypoint.

### Launch

```shell
cd FlagScale
python -m flagscale.cli serve qwen_gr00t -c ./examples/qwen_gr00t/conf/serve.yaml
```

Set `engine_args.model` to the checkpoint step directory that directly contains `pretrained_model`, for example `/path/to/outputs/.../checkpoints/030000` or `/path/to/outputs/.../checkpoints/last`.

### Required Config

The `qwen_gr00t` serve config declares:

- `engine_args.protocol`: websocket/msgpack request and response keys
- `engine_args.rename_map`: canonical observation keys to internal Qwen-GR00T observation keys
- `engine_args.task_key`: internal task key
- `engine_args.image_hw`: resize target `[H, W]`

Example:

```yaml
engine_args:
  model_variant: QwenGr00t
  model: /abs/path/to/checkpoint
  device: cuda
  image_hw: [224, 224]
  protocol:
    env_name: starvla_sim
    image_key: observation/image
    wrist_image_key: observation/wrist_image
    state_key: observation/state
    prompt_key: prompt
    actions_key: actions
  rename_map:
    observation.image: observation.images.image
    observation.wrist_image: observation.images.wrist_image
    observation.state: observation.state
  task_key: task
```

### Protocol

The example protocol matches the starVLA simulation websocket contract:

- request keys: `observation/image`, `observation/wrist_image`, `observation/state`, `prompt`, optional `observation/image_right`
- response key: `actions`

The default Qwen-Gr00t example is configured for the two-view checkpoint validated in local GPU testing. If your checkpoint expects a third image input, add `protocol.right_image_key` and the corresponding `rename_map` entry for `observation.image_right`.

The runner validates the configured protocol keys, converts the request into canonical observation keys, applies `rename_observations_processor` through preprocessor overrides, resizes images to `engine_args.image_hw`, and returns:

```python
{"actions": [[...], [...], ...]}
```

### Minimal Request Example

```python
import asyncio

import numpy as np
import websockets

from flagscale.serve import msgpack_numpy


async def main():
    async with websockets.connect("ws://127.0.0.1:5000", proxy=None) as websocket:
        metadata = msgpack_numpy.unpackb(await websocket.recv())
        print("metadata:", metadata)

        request = {
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros((8,), dtype=np.float32),
            "prompt": "Open the drawer",
        }

        await websocket.send(msgpack_numpy.packb(request))
        response = msgpack_numpy.unpackb(await websocket.recv())
        print(response["actions"])


asyncio.run(main())
```

The response is a msgpack-encoded dict that always includes `actions`. The server may also include extra fields such as `server_timing`.
