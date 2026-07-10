# DeepSeek V4 Flash DSpark — Abliterated · 2× DGX Spark

Serves the abliterated (uncensored) build of DeepSeek-V4-Flash-DSpark across two NVIDIA DGX Spark (GB10) nodes connected via InfiniBand/RoCE, using vLLM with DSpark speculative decoding, nvFP4 MLA KV-cache, and B12X MoE kernels.

---

## Prerequisites

- **2× NVIDIA DGX Spark (GB10)** linked via InfiniBand/RoCE
- **Docker** with GPU support on both machines
- **`hf` CLI** ([install guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)) — logged in with `hf login`
- **Passwordless SSH** from the master node to the worker node
- **`rsync`** installed on both machines
- **`~/gpu-clear.sh`** on both nodes (optional — failure is non-fatal)

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

| Variable | Example | Description |
|---|---|---|
| `MASTER` | `10.0.0.1` | IP of the master node (rank 0 — runs the API server) |
| `WORKER_ADDR` | `10.0.0.2` | IP of the worker node (rank 1 — headless compute) |
| `PORT` | `25000` | TCP port for vLLM multi-node control/store |
| `HCA` | `rocep1s0f1` | InfiniBand HCA device name |
| `IF` | `enp1s0f1np1` | Network interface for NCCL/GLOO/TP sockets |

These values must match your physical network. The publish cluster used `10.100.10.x` — if deploying there, update accordingly.

---

## `start.sh` — Cluster launch

A single script that orchestrates the entire cluster from the master node. No need to SSH to the worker or run anything on the second machine.

### What it does, step by step

| Step | Action | Detail |
|---|---|---|
| **1** | Pull container image | `docker pull ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10` and tags it as `vllm-dspark-runtime:dspark-nvfp4-stage-c` |
| **2** | Sync image to worker | Exports the image as a tar via `docker save` and pipes it through SSH to `docker load` on the worker — downloads once, uses on both |
| **3** | Download model weights | Runs `hf download drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored` to a local directory (`$MODELDIR`, default `~/models/dsv4-flash-dspark-abliterated`) |
| **4** | Sync model to worker | Uses `rsync` to copy the model to the exact same path on the worker node |
| **5** | Launch worker container | SSHes into the worker, clears GPU memory, starts the vLLM container in **rank 1** mode (`--headless` — compute only, no API) |
| **6** | Launch master container | Clears GPU locally, starts the vLLM container in **rank 0** mode (API server on port 8888, full inference endpoint) |
| **7** | Wait for readiness | Streams the container logs live to your terminal, then polls the `/health` endpoint silently. When the model responds, prints **"Model is ready!"** and returns control |

### Multi-node coordination

vLLM distributes across the two nodes using a TCP store:

- Rank 0 (master) creates the store at `--master-addr:--master-port`
- Rank 1 (worker) connects to that same address
- Both pass `--nnodes 2` with their respective `--node-rank`
- NCCL, Gloo, and TP communication happen over the InfiniBand interface (`$IF`) using the RoCE HCA (`$HCA`)

### Overridable environment variables

| Variable | Default | Description |
|---|---|---|
| `MODELDIR` | `$HOME/models/dsv4-flash-dspark-abliterated` | Where to store/sync the model weights |
| `SERVE_PORT` | `8888` | API port exposed on the master node |

---

## `stop.sh` — Cluster stop

Stops and removes the container on both nodes:

```bash
./stop.sh
```

It reads `WORKER_ADDR` from `.env`, then runs `docker stop` + `docker rm` on the worker (via SSH) and locally.

---

## Rebuilding the abliteration (optional)

The weights in this repo were produced by projecting a refusal-direction vector onto specific layers of the base model. You can rebuild or customise this process:

```bash
# After capturing refusal directions
# (see scripts/prompts.py for the prompt battery,
#  scripts/compute_direction.py for direction extraction)
python3 scripts/project_wob.py \
  --src ~/models/dsv4-flash-dspark \
  --dst ~/models/dsv4-flash-dspark-abliterated \
  --direction work/refusal_direction_r1.pt \
  --lambda-attn 3.5 --min-layer 10 --max-layer 42 --n-directions 1
```

The abliteration scripts:

| Script | Purpose |
|---|---|
| `scripts/prompts.py` | Prompt battery used to elicit refusals from the base model |
| `scripts/compute_direction.py` | Computes the refusal direction vector from model activations |
| `scripts/project_wob.py` | Projects the direction out of specific weight layers |
| `scripts/hybrid_overlay.py` | Layer-range hybrid overlay tooling |

The current weights use a hybrid layer-range strategy: layers 0–9 keep stock `attn.wo_b` (preserving chat/tools/protocol behaviour), layers 10–42 + MTP heads are abliterated, with SRA-cleaned rank-1 direction at λ=3.5.

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
