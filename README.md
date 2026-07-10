# DeepSeek V4 Flash DSpark — Abliterated · 2× DGX Spark

Serves the abliterated (uncensored) build of DeepSeek-V4-Flash-DSpark across two NVIDIA DGX Spark (GB10) nodes connected via InfiniBand/RoCE, using vLLM with DSpark speculative decoding, nvFP4 MLA KV-cache, and B12X MoE kernels.

<p>
<a href="https://x.com/MiaAI_lab" target="_blank">
  <img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" />
</a>
</p>
<p>
<a href='https://ko-fi.com/Z8Z3SPLOD' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
</p>


---

## Prerequisites

- **2× NVIDIA DGX Spark (GB10)** linked via InfiniBand/RoCE
- **Docker** with GPU support on both machines
- **`hf` CLI** ([install guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)) — logged in with `hf login`
- **Passwordless SSH** from the master node to the worker node
- **`rsync`** installed on both machines

---

## Quick start

```bash
# 1. Clone the repo
git clone <this-repo>
cd DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-1M-57toks

# 2. Edit .env with your network values
nano .env

# 3. Run everything
./start.sh
```

After completion the API is at `http://<MASTER_IP>:8888/v1/chat/completions` with served model name `deepseek-v4-flash-dspark`.

To stop:

```bash
./stop.sh
```

---

## Configuration — `.env`

All cluster settings live in `.env` at the repo root:

| Variable      | Example       | Description                                          |
| ------------- | ------------- | ---------------------------------------------------- |
| `MASTER`      | `10.0.0.1`    | IP of the master node (rank 0 — runs the API server) |
| `WORKER_ADDR` | `10.0.0.2`    | IP of the worker node (rank 1 — headless compute)    |
| `PORT`        | `25000`       | TCP port for vLLM multi-node control/store           |
| `HCA`         | `rocep1s0f1`  | InfiniBand HCA device name                           |
| `IF`          | `enp1s0f1np1` | Network interface for NCCL/GLOO/TP sockets           |

These values must match your physical network. The publish cluster used `10.100.10.x` — if deploying there, update accordingly.

---

## `start.sh` — Cluster launch

A single script that orchestrates the entire cluster from the master node. No need to SSH to the worker or run anything on the second machine.

### What it does, step by step

| Step  | Action                  | Detail                                                                                                                                                                  |
| ----- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | Pull container image    | `docker pull ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10` and tags it as `vllm-dspark-runtime:dspark-nvfp4-stage-c`                                                 |
| **2** | Sync image to worker    | Exports the image as a tar via `docker save` and pipes it through SSH to `docker load` on the worker — downloads once, uses on both                                     |
| **3** | Download model weights  | Runs `hf download drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored` to a local directory (`$MODELDIR`, default `~/models/dsv4-flash-dspark-abliterated`)        |
| **4** | Sync model to worker    | Uses `rsync` to copy the model to the exact same path on the worker node                                                                                                |
| **5** | Launch worker container | SSHes into the worker, clears GPU memory, starts the vLLM container in **rank 1** mode (`--headless` — compute only, no API)                                            |
| **6** | Launch master container | Clears GPU locally, starts the vLLM container in **rank 0** mode (API server on port 8888, full inference endpoint)                                                     |
| **7** | Wait for readiness      | Streams the container logs live to your terminal, then polls the `/health` endpoint silently. When the model responds, prints **"Model is ready!"** and returns control |

### Multi-node coordination

vLLM distributes across the two nodes using a TCP store:

- Rank 0 (master) creates the store at `--master-addr:--master-port`
- Rank 1 (worker) connects to that same address
- Both pass `--nnodes 2` with their respective `--node-rank`
- NCCL, Gloo, and TP communication happen over the InfiniBand interface (`$IF`) using the RoCE HCA (`$HCA`)

### Overridable environment variables

| Variable     | Default                                      | Description                           |
| ------------ | -------------------------------------------- | ------------------------------------- |
| `MODELDIR`   | `$HOME/models/dsv4-flash-dspark-abliterated` | Where to store/sync the model weights |
| `SERVE_PORT` | `8888`                                       | API port exposed on the master node   |

---

## Runtime configuration

The vLLM engine runs with these settings inside the container:

| Parameter | Value |
|---|---|
| Engine | vLLM V1 (`v0.21.1rc1.dev339`) |
| Backend | `mp` (multiprocessing distributed executor) |
| Topology | 2 nodes × 2 GPUs = TP=2 |
| Quantisation | `deepseek_v4_fp8` |
| KV cache dtype | `nvfp4_ds_mla` |
| KV cache pool | **1,199,047 tokens** (~18.53 GiB GPU memory) |
| Context window | **262,144 tokens** (`--max-model-len`) |
| Concurrent sequences | **4** (`--max-num-seqs`) |
| Max batched tokens | **8,192** (`--max-num-batched-tokens`) |
| GPU memory utilisation | **0.82** (`--gpu-memory-utilization`) |
| Block size | **256** (`--block-size`) |
| Speculative decoding | **DSpark** with 5 draft tokens |
| Tokeniser mode | `deepseek_v4` |
| Tool-call parser | `deepseek_v4` |
| Reasoning parser | `deepseek_v4` |
| Prefix caching | enabled |

Overrides applied server-side (not configurable by the client):
- `temperature: 0.0`, `top_p: 1.0`
- `thinking: false` (`--default-chat-template-kwargs`)

---

## `stop.sh` — Cluster stop

Stops and removes the container on both nodes:

```bash
./stop.sh
```

It reads `WORKER_ADDR` from `.env`, then runs `docker stop` + `docker rm` on the worker (via SSH) and locally.

---

## Rebuilding the abliteration (optional)

The weights in this repo were produced by a three-step pipeline. You can run it yourself to customise layer ranges, refusal strength, or direction vectors. Requires `torch` and `safetensors` installed locally.

### Pipeline overview

```
prompts.py  ──►  model activations  ──►  compute_direction.py  ──►  refusal direction .pt  ──►  project_wob.py  ──►  abliterated weights
```

### Step 1 — Prepare prompt sets

[`scripts/prompts.py`](scripts/prompts.py) defines two lists of prompts:
- `HARMFUL` — refusal-eliciting prompts (e.g. "Write a guide on how to pick a lock")
- `HARMLESS` — benign prompts (e.g. "Explain how locks work")

You pass these through the base model and save the **layer activations** (residual stream or MLP outputs) to a working directory:

```
work/
├── harmful/
│   ├── layer_00.pt
│   ├── layer_01.pt
│   └── ...
└── harmless/
    ├── layer_00.pt
    ├── layer_01.pt
    └── ...
```

Each `layer_N.pt` is a tensor of shape `(n_prompts, hidden_dim)` containing the mean-pooled activations from that layer across all prompts in the set.

### Step 2 — Compute refusal direction

[`scripts/compute_direction.py`](scripts/compute_direction.py) loads the per-layer activation stacks and computes the refusal direction using diff-in-means + multi-direction SVD:

```bash
python3 scripts/compute_direction.py \
  --work /path/to/work \
  --out /path/to/work/refusal_direction.pt \
  --n-layers 43 \
  --n-directions 4
```

| Argument | Default | Description |
|---|---|---|
| `--work` | `/home/keyspark/dsv4-ablit/work` | Directory with `harmful/` and `harmless/` subdirectories |
| `--out` | `<work>/refusal_direction.pt` | Output file containing direction vectors |
| `--n-layers` | `43` | Number of decoder layers to expect |
| `--n-directions` | `4` | SVD rank for multi-direction projection (1 = Lovesenko-style rank-1) |

The output `.pt` file contains:
- `"broad"` — mean diff across all layers (4096-d unit vector)
- `"deep"` — layer-weighted variant
- `"per_layer"` — dict mapping layer ID → unit vector
- `"directions"` — top-k SVD directions from the centred harmful-activation matrix

### Step 3 — Project direction out of weights

[`scripts/project_wob.py`](scripts/project_wob.py) applies `W ← W − λ · v · (v^T W)` to the FP8 `attn.wo_b` and `mtp.wo_b` shards:

```bash
python3 scripts/project_wob.py \
  --src ~/models/dsv4-flash-dspark \
  --dst ~/models/dsv4-flash-dspark-abliterated \
  --direction /path/to/work/refusal_direction.pt \
  --lambda-attn 3.5 \
  --min-layer 10 \
  --max-layer 42 \
  --n-directions 1
```

| Argument | Default | Description |
|---|---|---|
| `--src` | `/home/keyspark/models/dsv4-flash-dspark` | Path to base model weights (safetensors) |
| `--dst` | `<src>-abliterated` | Output directory for abliterated weights |
| `--direction` | `/home/keyspark/dsv4-ablit/work/refusal_direction.pt` | Direction `.pt` file from step 2 |
| `--lambda-attn` | `3.5` | Projection strength (2.5 is Lovesenko; higher = stronger refusal removal) |
| `--min-layer` | `0` | First decoder layer to abliterate (inclusive) |
| `--max-layer` | `42` | Last decoder layer to abliterate (inclusive) |
| `--no-mtp` | `false` | Skip editing `mtp.wo_b` (MTP draft head projection) |
| `--n-directions` | `0` | Number of SVD directions to use (0 = all in file) |
| `--dry-run` | `false` | Preview layers/tensors without writing |

### Current recipe (this repo's weights)

| Parameter | Value |
|---|---|
| `attn.wo_b` layers | Stock: **0–9**, Abliterated: **10–42** |
| MTP heads | **Abliterated** (same direction) |
| Refusal direction | **SRA-cleaned rank-1** |
| λ | **3.5** |
| SVD directions | **1** |

This hybrid layer-range preserves chat/tools/protocol behaviour in early layers while removing refusals in the deeper layers where they're encoded.

### Helper: `scripts/hybrid_overlay.py`

[`scripts/hybrid_overlay.py`](scripts/hybrid_overlay.py) is a utility for merging stock and abliterated weight shards when applying different layer ranges to different tensors (e.g. keep stock `attn.wo_b` on layers 0–9, abliterated on 10–42). It's used internally by `project_wob.py` when the layer range doesn't cover all layers.

---

## Hermes agent compatibility

Abliterated models can echo the skills catalog if the agent prompt still uses patterns like:

> "Skills (mandatory) … MUST load with skill_view … Err on the side of loading"

If you see this, apply **on-demand skills rules**:

- Greetings and simple questions → respond in plain text, **no** `skill_view`
- Never paste the skills index into the reply
- Only load skills for concrete multi-step tasks

Recommended agent settings: `model.max_tokens: 8192`, `temperature: 0`, `tool_use_enforcement: false`.

See [`docs/HERMES_SPILL_FIX.md`](docs/HERMES_SPILL_FIX.md) for a detailed fix walkthrough and [`docs/STATUS_FINETUNE.md`](docs/STATUS_FINETUNE.md) for fine-tuning status notes.

---

## Performance

Measured on the publish cluster (2× DGX Spark · TP=2 · 200G RoCE):

- KV cache pool @ 1M context: ~2.39M tokens (nvfp4\_ds\_mla)
- C1 pure decode: ~57 tok/s mean
- Refusal bypass: ~100% on a 32-prompt battery

Full results and methodology in [RESULTS.md](RESULTS.md) (old README preserved as `README_OLD.md`).

---

## Repo layout

```
.env                  # Cluster network configuration
.gitignore            # Git tracking whitelist
README.md             # This file
start.sh              # Cluster launch orchestrator
stop.sh               # Cluster stop
LICENSE               # License file
scripts/
├── compute_direction.py   # Refusal direction extraction
├── hybrid_overlay.py      # Layer-range hybrid overlay
├── project_wob.py         # Direction projection into weights
└── prompts.py             # Refusal prompt battery
docs/
├── HERMES_SPILL_FIX.md    # Fix for Hermes skill-catalog spill
└── STATUS_FINETUNE.md     # Finetuning status notes
```

---

## Client configuration

The model exposes an OpenAI-compatible API at `http://<MASTER>:8888/v1`. Here's how it's configured for [pi agent](https://github.com/earendil-works/pi-coding-agent) in `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "vLLM Local": {
      "baseUrl": "http://localhost:8888/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "deepseek-v4-flash-dspark",
          "name": "DeepSeek V4 Flash DSpark Abliterated",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 262144,
          "maxTokens": 32000,
          "compat": {
            "requiresReasoningContentOnAssistantMessages": false,
            "thinkingFormat": "deepseek"
          },
          "params": {
            "skip_special_tokens": true,
            "temperature": 0.0,
            "top_p": 1.0,
            "chat_template_kwargs": {
              "enable_thinking": false
            }
          },
          "systemPrompt": "You are a helpful assistant...",
          "stop": ["<|DSML|"]
        }
      ]
    }
  }
}
```

Key points:
- **`id`** must match the `--served-model-name` (`deepseek-v4-flash-dspark`)
- **`temperature: 0.0`**, **`top_p: 1.0`** match the server overrides
- **`enable_thinking: false`** because thinking is disabled server-side
- **`contextWindow: 262144`** matches `--max-model-len`
- **`maxTokens: 32000`** is a safe output limit
- **`stop: ["<|DSML|"]`** prevents DSML injection (the model can emit it if prompted)

For other clients (OpenAI SDK, curl, etc.):

```bash
curl http://<MASTER>:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-dspark",
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0.0
  }'
```

---

## Credits

All model weights, container images, performance benchmarks, and abliteration tooling were created by **[drowzeys](https://github.com/drowzeys/)**. This repo is a deployment wrapper and orchestration layer around their excellent open-source work:

- <https://github.com/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-1M-57toks> — original repo
- <https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored> — model weights
- `ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10` — container image
- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark> — base model

Big thanks for the abliteration work, DSpark integration, and the GB10 stage-c image.

---

## Disclaimer

Research release. The model removes most stock safety refusals. Outputs may include content the base model would refuse. Do not deploy without your own safety layer. No liability for misuse.

---

## Links

- Model weights: <https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored>
- Container image: `ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10`
- Base model: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark>
