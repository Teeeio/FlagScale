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

The unified VLA serving entrypoint is `flagscale/serve/run_serve_vla.py`. It is wired into the `qwen_gr00t` serve example and accepts the starVLA simulation websocket protocol.

### Recommended Launch Path

Use the example config:

```shell
cd FlagScale
python -m flagscale.cli serve qwen_gr00t -c ./examples/qwen_gr00t/conf/serve.yaml
```

The default `engine_args.model` in the example config is a placeholder path. Replace it
with your actual checkpoint directory before serving, or pass it with `--model-path`.
The path must point to the checkpoint step directory that directly contains
`pretrained_model`, for example `/path/to/outputs/.../checkpoints/030000` or
`/path/to/outputs/.../checkpoints/last`.

You can override the checkpoint path and runtime fields from the command line:

```shell
cd FlagScale
python -m flagscale.cli serve qwen_gr00t \
  -c ./examples/qwen_gr00t/conf/serve.yaml \
  --model-path /abs/path/to/checkpoint \
  --port 5000 \
  --engine-args '{"device":"cuda","missing_image_policy":"error"}'
```

If your shell-level `flagscale` command does not show `-c/--config` in `serve --help`,
it is likely resolving a different package entrypoint. In that case, keep using
`python -m flagscale.cli ...` from the FlagScale repo root, or reinstall this repo's
CLI entrypoint with `pip install -e .` in the environment where you run FlagScale.

The runtime GPU selection is controlled by `experiment.envs.CUDA_VISIBLE_DEVICES`
in `examples/qwen_gr00t/conf/serve.yaml`.

### Direct Script Launch

If you want to invoke the entrypoint directly:

```shell
cd FlagScale
export PYTHONPATH=$(pwd):$PYTHONPATH
python flagscale/serve/run_serve_vla.py \
  --config-path /abs/path/to/serve_runtime.yaml \
  --log-dir /tmp/flagscale_serve_logs
```

The direct script expects a YAML file with a top-level `serve:` list, for example:

```yaml
serve:
  - serve_id: local_vla
    engine_args:
      host: 0.0.0.0
      port: 5000
      model_variant: QwenGr00t
      model: /abs/path/to/checkpoint
      device: cuda
      missing_image_policy: error
```

### Runtime Arguments

Common `engine_args` for the VLA runner:

- `model_variant`: currently use `QwenGr00t`
- `model`: checkpoint directory
- `device`: runtime device such as `cuda`
- `host`: bind address, default `0.0.0.0`
- `port`: websocket port, default `5000`
- `missing_image_policy`: `error` or `black`, default `error`

Optional `engine_args`:

- `image_hw`: override auto-resolved resize target as `[H, W]`
- `state_key`: override the resolved internal state feature key
- `starvla_slot_map`: explicitly map starVLA visual slots to internal visual feature keys

### Request And Response Protocol

The websocket request follows the starVLA simulation key layout:

- `observation/image`
- `observation/wrist_image`
- `observation/state`
- `prompt`
- optional `observation/image_right`

The server response is:

```python
{"actions": [[...], [...], ...]}
```

The runner automatically:

- remaps external starVLA keys to internal model feature keys using checkpoint/model features when available
- resizes images before the checkpoint preprocessor runs
- preserves the checkpoint pre/postprocessor pipeline

### Current Support Scope

At the moment, this serving path only supports `QwenGr00t`.

Other VLA models are not declared supported yet and will be added separately.

For Qwen-GR00T checkpoints, both common 2-view and 3-view naming patterns are handled by the automatic starVLA slot inference logic.
