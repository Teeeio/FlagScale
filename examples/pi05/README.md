# Install FlagScale

Clone FlagScale code from github.

If you don't have access to the international internet, import FlagScale project on [gitee](https://gitee.com/), then clone from gitee.

```sh
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale/
```

Install train and inference env according to [README](https://github.com/FlagOpen/FlagScale/blob/main/README.md).

For a fresh machine, you can create the environments with:

```sh
./install/install-requirements.sh --env train
./install/install-requirements.sh --env inference --llama-cpp-backend cuda
```

Pi0.5 training uses the `flagscale-train` conda env.

# Download Model

OpenPI Pi0.5 base weights (original source):

```sh
git lfs install
mkdir -p /share/pi05_models
git clone https://huggingface.co/lerobot/pi05_base /share/pi05_models/openpi05_base
```

If you don't have access to the international internet, download from ModelScope:

```sh
pip install modelscope
mkdir -p /share/pi05_models/openpi05_base
modelscope download --model heart6/openpi05_base --local_dir /share/pi05_models/openpi05_base
```

Tokenizer file for language tokens:
`/share/pi05_models/openpi05_base/paligemma_tokenizer.model`

If it is missing, download the tokenizer only.

Hugging Face:

```sh
wget -O /share/pi05_models/openpi05_base/paligemma_tokenizer.model \
  https://huggingface.co/google/paligemma-3b-pt-224/resolve/main/tokenizer.model
```

ModelScope:

```sh
modelscope download --model google/paligemma-3b-pt-224 \
  --include tokenizer.model \
  --local_dir /share/pi05_models/openpi05_base
mv /share/pi05_models/openpi05_base/tokenizer.model \
  /share/pi05_models/openpi05_base/paligemma_tokenizer.model
```

## Convert Weights

Convert the OpenPI checkpoint into FlagScale format. This is required for training/inference in FlagScale.
Run this in the `flagscale-train` environment.

This step requires JAX/Orbax/Flax. Install them (CPU is sufficient for conversion):

```sh
pip install -r requirements/requirements-pi05-convert.txt
```

```sh
JAX_PLATFORMS=cpu PYTHONPATH=./:$PYTHONPATH python tools/convert_openpi_pi05_to_flagscale.py \
  --checkpoint_dir /share/pi05_models/openpi05_base \
  --output_path /share/pi05_models/openpi05_base/flagscale_pi05.safetensors \
  --export_vlm \
  --vlm_output_dir /share/pi05_models/openpi05_base_pytorch \
  --vlm_precision float32
```

This also exports fp32 VLM weights for SigLIP, used in training and inference.

# Download Statistics

The stats file is included in the OpenPI assets:
`/share/pi05_models/openpi05_base/assets/physical-intelligence/libero/norm_stats.json`

This file contains normalization stats (mean/std and optional quantiles) for state/action scaling.
No extra download is required once the model assets are available.

If it is missing, download the stats file only.

Hugging Face:

```sh
wget -O /share/pi05_models/openpi05_base/assets/physical-intelligence/libero/norm_stats.json \
  https://huggingface.co/lerobot/pi05_base/resolve/main/assets/physical-intelligence/libero/norm_stats.json
```

ModelScope:

```sh
modelscope download --model heart6/openpi05_base \
  --include assets/physical-intelligence/libero/norm_stats.json \
  --local_dir /share/pi05_models/openpi05_base
```

# Training

## Prepare Dataset

Install a dataset loader (lerobot preferred, datasets is the fallback):

```sh
pip install lerobot
# or
pip install datasets
```

Only use the official dataset:

```sh
mkdir -p /share/pi05_datasets/Rhx11111/pi05_libero_data
modelscope download --dataset Rhx11111/pi05_libero_data --local_dir /share/pi05_datasets/Rhx11111/pi05_libero_data
```

Extract the training subset (smaller chunked set) and use it as the dataset root:

```sh
tar -xzf /share/pi05_datasets/Rhx11111/pi05_libero_data/libero_1_head_cam_only_1000_chunk.tar.gz \
  -C /share/pi05_datasets/Rhx11111/pi05_libero_data
```

Dataset root:
`/share/pi05_datasets/Rhx11111/pi05_libero_data/libero_1_head_cam_only_1000_chunk`

Verify the dataset follows OpenPI conversion (check `data/chunk-000/` and `meta/info.json`).

## Edit Config

```sh
cd FlagScale/
vim examples/pi05/conf/train/pi05.yaml
```

Change 8 fields:
- model.checkpoint_dir -> /share/pi05_models
- model.pretrain_weights -> /share/pi05_models/openpi05_base/flagscale_pi05.safetensors
- model.siglip_weights -> /share/pi05_models/openpi05_base_pytorch/model.safetensors
- data.data_path -> /share/pi05_datasets/Rhx11111/pi05_libero_data/libero_1_head_cam_only_1000_chunk
- data.stat_path -> /share/pi05_models/openpi05_base/assets/physical-intelligence/libero/norm_stats.json
- data.tokenizer_path -> /share/pi05_models/openpi05_base/paligemma_tokenizer.model
- data.state_dim -> 8
- data.num_state_bins -> 256

## Start Training

```sh
cd FlagScale/
python run.py --config-path ./examples/pi05/conf --config-name train action=run
```

## Multi-GPU Template (Optional)

The multi-GPU template is under `examples/pi05/conf/train/variants/pi05_multi.yaml`.
Pass the GPU count as a parameter when launching.

```sh
CUDA_VISIBLE_DEVICES=<gpu_ids> \
python run.py --config-path ./examples/pi05/conf --config-name train \
  train=variants/pi05_multi experiment.runner.nproc_per_node=<num_gpus> action=run
```

# Inference

## Edit Config

```sh
cd FlagScale/
vim examples/pi05/conf/inference/pi05.yaml
```

Change 5 fields:
- engine.model -> /share/pi05_models/openpi05_base/flagscale_pi05.safetensors
- engine.stat_path -> /share/pi05_models/openpi05_base/assets/physical-intelligence/libero/norm_stats.json
- engine.tokenizer -> /share/pi05_models/openpi05_base/paligemma_tokenizer.model
- engine.pi05_siglip_weights -> /share/pi05_models/openpi05_base_pytorch/model.safetensors
- engine.num_state_bins -> 256

## Start Inference

```sh
cd FlagScale/
python run.py --config-path ./examples/pi05/conf --config-name inference action=run
```

## Optional: OpenPI-Aligned SigLIP Inference

If you need OpenPI-aligned SigLIP vision tokens, use the dedicated entrypoint:

```sh
cd FlagScale/
python flagscale/inference/embodied_entrypoint_pi05.py \
  --config-path examples/pi05/conf/inference/pi05.yaml
```

In the config, add:
- engine.pi05_siglip_weights -> /share/pi05_models/openpi05_base_pytorch/model.safetensors
- engine.pi0_model, engine.pi0_tokenizer, engine.pi0_stat if you want Pi0 language tokens

# Serving

## Edit Config

```sh
cd FlagScale/
vim examples/pi05/conf/serve/pi05.yaml
```

Change 5 fields:
- engine.model -> /share/pi05_models/openpi05_base/flagscale_pi05.safetensors
- engine.stat_path -> /share/pi05_models/openpi05_base/assets/physical-intelligence/libero/norm_stats.json
- engine.tokenizer -> /share/pi05_models/openpi05_base/paligemma_tokenizer.model
- engine.pi05_siglip_weights -> /share/pi05_models/openpi05_base_pytorch/model.safetensors
- engine.num_state_bins -> 256

## Run Serving

```sh
cd FlagScale/
python run.py --config-path ./examples/pi05/conf --config-name serve action=run
```

# Test Server with Client

Download test images:

```sh
cd FlagScale/
wget https://gitee.com/hchnr/flag-scale/raw/robotics_dataset/orbbec_0_latest.jpg
wget https://gitee.com/hchnr/flag-scale/raw/robotics_dataset/orbbec_1_latest.jpg
wget https://gitee.com/hchnr/flag-scale/raw/robotics_dataset/orbbec_2_latest.jpg
```

Run client:

```sh
cd FlagScale/
python examples/pi05/client_pi05.py \
  --host 127.0.0.1 \
  --port 5000 \
  --base-img orbbec_0_latest.jpg \
  --left-wrist-img orbbec_1_latest.jpg \
  --right-wrist-img orbbec_2_latest.jpg \
  --state-dim 8 \
  --action-dim 32 \
  --action-horizon 10 \
  --instruction "Pick up the red object and place it in the container."
```

# Pi0.5 Specific Features

- State discretization: 8D state -> 256-bin discrete tokens
- Flow matching for action prediction
- AdaRMSNorm blocks
