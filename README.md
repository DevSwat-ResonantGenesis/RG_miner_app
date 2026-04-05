# RG Miner App

**Standalone mining client for the ResonantGenesis decentralized LLM training network.**

Download, login, and start earning **$RGT tokens** by training AI models on your GPU. Your machine becomes a node in a global pipeline-parallel training swarm.

> **Open source under AGPL-3.0** — every line of training logic, model architecture, gradient compression, and blockchain recording is auditable in this repo and the [server-side repos](https://github.com/DevSwat-ResonantGenesis).

---

## Quick Start

### macOS

```bash
# Prerequisites (needed once — for P2P WebRTC)
brew install ffmpeg pkg-config

# Setup
git clone https://github.com/DevSwat-ResonantGenesis/RG_miner_app.git
cd RG_miner_app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

### Linux (Ubuntu/Debian)

```bash
# Prerequisites (needed once — for P2P WebRTC)
sudo apt install ffmpeg libavdevice-dev pkg-config

# Setup
git clone https://github.com/DevSwat-ResonantGenesis/RG_miner_app.git
cd RG_miner_app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

### Windows

```bash
git clone https://github.com/DevSwat-ResonantGenesis/RG_miner_app.git
cd RG_miner_app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

> **Windows note:** You may need to install ffmpeg separately for P2P WebRTC. Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

Then open **http://localhost:3000** and login with your [ResonantGenesis account](https://dev-swat.com).

> **Important:** On macOS, the command is `python3` and `pip` (after activating the venv). If you see `zsh: command not found: python`, use `python3` instead. Once inside an activated venv, both `python` and `pip` work.

### CUDA Support (NVIDIA GPUs)

If you have an NVIDIA GPU and want CUDA acceleration:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Run this **before** `pip install -r requirements.txt`, or it will install CPU-only PyTorch.

---

## What This Actually Does

We know "train AI on your GPU + earn crypto" raises eyebrows. Here's the transparent, code-backed explanation.

### Real GPU Training

This is not fake mining or idle hashing. The miner runs actual **PyTorch forward and backward passes** on a transformer model. You can see the full model definition in [`model_architecture.py`](model_architecture.py) — it's a standard transformer with Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), and SwiGLU activations. Same architecture techniques used in LLaMA and Mistral.

The training engine is in [`real_trainer.py`](real_trainer.py) — tokenizer, data loading, loss computation, gradient compression via Top-K sparsification (keeping only the top 0.01% of gradient values), and SHA-256 hash verification of every gradient submission.

### Pipeline-Parallel Architecture

Large models (7B-405B parameters) don't fit on a single consumer GPU. The network splits models into **layer groups** across multiple miners using pipeline parallelism:

- Each miner holds a **shard** (e.g., layers 0-12 of a 24-layer model)
- Training uses **1F1B microbatch scheduling** (same technique as Microsoft DeepSpeed / Meta pipeline training)
- Activations flow forward between stages, gradients flow backward
- With 32 microbatches and 4 stages: **91% pipeline efficiency**

The 1F1B engine is in [`microbatch_engine.py`](microbatch_engine.py) with [12/12 tests passing](test_microbatch_engine.py). The model sharding logic is in [`moe_architecture.py`](moe_architecture.py).

```
Time -->
Stage 0:  F0 F1 F2 F3 B0 F4 B1 F5 B2 B3 B4 B5
Stage 1:     F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 B4 B5
Stage 2:        F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 B5
Stage 3:           F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5

F = forward microbatch, B = backward microbatch
```

### P2P Weight Transfer via WebRTC

Miners behind home routers (NAT) discover each other through **WebRTC signaling** — no manual port forwarding needed. The server-side `P2PDiscovery` service handles ICE candidate exchange and establishes DataChannels for direct miner-to-miner communication.

Weight shards are downloaded directly from peer miners via `/p2p/serve-weights`. If no peers have your shard, it falls back to the genesis seed. When miners disconnect, **liquid redistribution** automatically reassigns shards to healthy miners.

### Gradient Verification

Every gradient submission is verified:
1. **Top-K compression** — only the top 0.01% of gradient values are transmitted (100x compression)
2. **SHA-256 hash** — gradient integrity verified on both sides
3. **Staleness check** — gradients from outdated model versions are rejected
4. **Parameter server validation** — the aggregation server verifies each contribution before crediting rewards

---

## $RGT Token Economics

### How Rewards Work

Every accepted gradient submission earns $RGT. The amount depends on your miner tier:

| Tier | Class | Reward per Gradient | Multiplier |
|------|-------|-------------------|------------|
| Genesis Validator (T1) | `validator_miner` | 150 $RGT | 1.5x |
| Core Contributor (T2) | `core_miner` | 125 $RGT | 1.25x |
| Standard Miner (T3/T4) | `miner` | 100 $RGT | 1.0x |

### Halving Schedule

Block rewards halve yearly, similar to Bitcoin:

| Year | Block Reward |
|------|-------------|
| 1 | 100 $RGT |
| 2 | 50 $RGT |
| 3 | 25 $RGT |
| 4 | 12.5 $RGT |

### A Closed Utility Economy

**$RGT is not trying to be traded on exchanges.** It's a platform currency:

- **Earn** by contributing GPU compute to the training network
- **Spend** on IDE access, LLM API calls, agent creation and management on the [ResonantGenesis platform](https://dev-swat.com)

That's a closed utility loop — value comes from real platform usage, not exchange speculation. This is more sustainable than "earn tokens and hope someone buys them."

The real break-even question isn't "when does $RGT hit X price on a DEX" — it's **"does my mining output cover what I'd otherwise pay in platform fees?"** If you're using the Resonant IDE, LLM APIs, or AI agents, mining makes those services effectively free.

### On-Chain Recording

Every gradient submission and reward is recorded as a `training_gradient` transaction on the **ResonantGenesis Blockchain** (chain ID: `resonant-genesis-external-1`), which uses Raft consensus with Merkle-tree block validation. This creates an immutable provenance trail for all training contributions.

### Future Base L2 Anchor

$RGT currently lives on the ResonantGenesis sovereign chain. A cross-chain bridge to Base/ETH mainnet is planned as the network matures — we're building the utility first, not the speculation layer.

---

## Network Architecture

Your miner connects to a **3-service mesh**:

| Service | Role |
|---------|------|
| **Lighthouse** | P2P discovery, peer registry, pipeline topology management |
| **Mining** | Training orchestration, gradient aggregation, reward distribution |
| **External Blockchain** | Raft consensus, block production, on-chain transaction recording |

```
You (Miner)                    Mining Service                  Peer Miners
    |                               |                               |
    |-- 1. Login ------------------>|                               |
    |-- 2. Register Capability ---->|                               |
    |<- 3. Shard Assignment --------|                               |
    |-- 4. Request Transfer Plan -->|                               |
    |<-------- 5. P2P Download -----------------------------------------|
    |-- 6. Report Weights Loaded -->|                               |
    |                               |                               |
    |== Training Loop ==============|                               |
    |<- 7. Receive Task ------------|                               |
    |-- 8. Train (1F1B pipeline) ---|-- Activations/Gradients ----->|
    |-- 9. Submit Gradient -------->|                               |
    |<- 10. $RGT Reward ------------|                               |
```

---

## Why Auth Is Required

Without identity verification, anyone could submit garbage gradients and claim rewards. The auth system ensures every gradient is tied to a verified account for quality control, slashing penalties, and fair reward distribution.

- **How it works:** Your credentials are sent to the platform auth service via HTTPS and you receive a JWT token stored locally. All subsequent API calls include this token.
- **Same account:** This is the same login used by the Resonant IDE — one account for the entire ResonantGenesis platform.
- **Security:** All services run behind HTTPS with HSTS, CORS lockdown, and fail-closed auth in production. No passwords are stored by the miner app.
- **No lock-in:** The `RG_PLATFORM_URL` is configurable. All backend service code is open-source under AGPL-3.0.

---

## About This Project

### Who Built This

[DevSwat-ResonantGenesis](https://github.com/DevSwat-ResonantGenesis) — a small team that shipped code before marketing it. The parent organization operates [dev-swat.com](https://dev-swat.com), which runs the Resonant IDE, AI agents, LLM APIs, Hash Sphere memory, and Code Visualizer.

### What You Can Verify Right Now

- **7+ repos on GitHub** — miner app, mining service, blockchain, lighthouse, crypto service, memory service, frontend. All open-source under AGPL-3.0.
- **Real ML engineering:** Raft consensus from scratch, 1F1B pipeline parallelism, GQA+RoPE+SwiGLU transformer, Top-K gradient compression with SHA-256 verification, WebRTC P2P NAT traversal, slashing with Merkle proof verification.
- **Production infrastructure:** Docker-composed microservices, Nginx TLS termination, JWT auth, HSTS, CORS lockdown.
- **Live platform:** [dev-swat.com](https://dev-swat.com) is live and running the services that $RGT pays for.

### What We Haven't Done Yet

We believe in being honest about where we are:

- **No Base mainnet token contract** — $RGT lives on our sovereign chain for now. The bridge comes when the network proves itself.
- **No third-party security audit** — the code is open for anyone to audit, but we haven't paid for a formal one yet.
- **No large miner network** — we need early participants to help stress-test the P2P pipeline.
- **Small community** — the repos are new. Every project starts somewhere.

### The Honest Pitch

If you're looking for a proven, battle-tested network — wait. If you want to be an early contributor to a technically real project, help the network form, and accumulate $RGT before the crowd shows up — that's what early participation looks like. We'd rather be transparent about where we are than fake momentum we don't have.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RG_PLATFORM_URL` | `https://dev-swat.com` | Platform base URL |
| `RG_MINER_PORT` | `3000` | Local dashboard port |
| `RG_MINING_CYCLES` | `100` | Training cycles per session |

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9+ | 3.11+ |
| RAM | 8 GB | 16+ GB |
| GPU | None (CPU fallback) | NVIDIA CUDA or Apple MPS |
| GPU VRAM | 4 GB (Seed 1B) | 24+ GB (larger models) |
| PyTorch | 2.1+ | 2.4+ |
| Network | Broadband internet | 100+ Mbps (for P2P weight transfer) |

## Project Structure

```
RG_miner_app/
├── server.py              # FastAPI backend — auth, mining loop, P2P endpoints, WebSocket dashboard
├── microbatch_engine.py   # 1F1B pipeline-parallel execution engine (warmup/steady/cooldown)
├── moe_architecture.py    # ModelShard — pipeline-parallel model slicing for sharded training
├── model_architecture.py  # Full PyTorch Transformer (GQA + RoPE + SwiGLU) for single-GPU mode
├── real_trainer.py        # Training engine, tokenizer, data loading, gradient compression
├── static/
│   └── index.html         # Single-page dashboard (vanilla JS, Chart.js, no build step)
├── test_microbatch_engine.py  # Tests for 1F1B engine (12/12 passing)
├── requirements.txt
├── LICENSE                # AGPL-3.0
└── README.md
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `command not found: python` | Use `python3` instead — macOS doesn't ship `python`. Once inside a venv, `python` works. |
| `command not found: pip` | Use `pip3`, or activate your venv first (`source venv/bin/activate`), then `pip` works. |
| `aiortc` / `PyAV` build error | Install system prerequisites first: `brew install ffmpeg pkg-config` (macOS) or `sudo apt install ffmpeg libavdevice-dev pkg-config` (Linux). Then re-run `pip install -r requirements.txt`. |
| `torch not found` | `pip install torch` — or see [pytorch.org](https://pytorch.org/get-started/locally/) for CUDA-specific install |
| `Connection refused` | Check that `RG_PLATFORM_URL` is reachable. The platform must be running. |
| `No tasks available` | The network may not have started a training epoch yet. Wait or check the dashboard. |
| `OOM on GPU` | The miner auto-scales batch size. For MPS/CPU it uses batch_size=1, seq_len=512. |
| `Pipeline training failed` | Falls back to standard training automatically. Check logs for P2P connectivity issues. |

## Contributing

Contributions welcome! This project is licensed under **AGPL-3.0** — any modifications or derivative works must also be open-sourced under the same license, including network use.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push and open a Pull Request

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

Copyright (c) 2026 DevSwat — ResonantGenesis

You are free to use, modify, and distribute this software under the terms of the AGPL-3.0. If you run a modified version of this software as a network service, you **must** make your source code available to users of that service.
