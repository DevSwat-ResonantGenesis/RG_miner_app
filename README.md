# RG Miner App ⛏️

**Standalone mining client for the ResonantGenesis decentralized LLM training network.**

Download, login, and start earning **$RGT tokens** by training AI models on your GPU. Your machine becomes a node in a global pipeline-parallel training swarm.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/DevSwat-ResonantGenesis/RG_miner_app.git
cd RG_miner_app

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the miner
python server.py

# 5. Open http://localhost:3000 and login with your ResonantGenesis account
```

That's it. Once you login and hit **Start Mining**, the app connects to the network, receives a shard assignment, downloads weights from peers, and begins training.

## Features

- **Platform Login** — Authenticate with your ResonantGenesis account (same as Resonant IDE)
- **Real GPU Training** — Actual PyTorch forward/backward passes on CUDA, MPS, or CPU
- **Pipeline-Parallel Training** — 1F1B microbatch scheduling across multi-GPU pipelines
- **P2P Weight Transfer** — Download model weights directly from peer miners (no central bottleneck)
- **Liquid Redistribution** — If a miner goes offline, shards auto-redistribute to healthy miners
- **Sharded Training** — Large models (7B–405B) split across multiple GPUs automatically
- **Model Registry** — View all available models from Seed 1B to Frontier 405B
- **Live Dashboard** — Real-time loss curves, reward tracking, training logs
- **Network Monitor** — See Lighthouse, Mining, and Blockchain service status
- **Gradient Compression** — Top-K compression (100x) for efficient gradient submission
- **$RGT Rewards** — Earn tokens for every accepted gradient

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9+ | 3.11+ |
| RAM | 8 GB | 16+ GB |
| GPU | None (CPU fallback) | NVIDIA CUDA or Apple MPS |
| GPU VRAM | 4 GB (Seed 1B) | 24+ GB (larger models) |
| PyTorch | 2.1+ | 2.4+ |
| Network | Broadband internet | 100+ Mbps (for P2P weight transfer) |

## How It Works

```
You (Miner)                    Mining Service                  Peer Miners
    |                               |                               |
    |-- 1. Login ------------------>|                               |
    |-- 2. Register Capability ---->|                               |
    |<- 3. Shard Assignment --------|                               |
    |-- 4. Request Transfer Plan -->|                               |
    |<-------- 5. P2P Download ------------------------------ ----->|
    |-- 6. Report Weights Loaded -->|                               |
    |                               |                               |
    |== Training Loop (repeats) ====|                               |
    |<- 7. Receive Task ------------|                               |
    |-- 8. Train (1F1B pipeline) ---|-- Activations/Gradients ----->|
    |-- 9. Submit Gradient -------->|                               |
    |<- 10. $RGT Reward ------------|                               |
```

### Detailed Flow

1. **Login** — Credentials sent to platform auth service, JWT token stored locally
2. **Register Capability** — Report your GPU model, VRAM, region to the Mining service
3. **Shard Assignment** — Server assigns you a slice of the model (e.g., layers 0-12 of a 24-layer model)
4. **Weight Transfer Plan** — Server tells you which peers already have your layers
5. **P2P Weight Download** — Pull weights directly from peers via `/p2p/serve-weights` (fallback: genesis seed)
6. **Report Loaded** — Tell the server your shard is ready for training
7. **Receive Task** — Get a training task (epoch, batch index, hyperparameters)
8. **Train** — Run 1F1B microbatch schedule (forward/backward passes interleaved for GPU efficiency)
9. **Submit Gradient** — Compressed gradient (Top-K with SHA256 hash) sent to parameter server
10. **Earn Reward** — $RGT tokens credited for accepted gradients

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RG_PLATFORM_URL` | `https://dev-swat.com` | Platform base URL |
| `RG_MINER_PORT` | `3000` | Local dashboard port |
| `RG_MINING_CYCLES` | `100` | Training cycles per session |

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

## Training Modes

### Full-Model Mode (default for small models)
Your GPU holds the entire model. Standard forward/backward on the full batch.

### Pipeline-Parallel Mode (automatic for large models)
The model is split across multiple miners. Each miner holds a **shard** (contiguous layers). Training uses 1F1B (One-Forward-One-Backward) microbatch scheduling:

```
Stage 0:  F0 F1 F2 F3 B0 F4 B1 F5 B2 B3 B4 B5   ← warmup then steady-state
Stage 1:     F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 B4 B5
Stage 2:        F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 B5
Stage 3:           F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5  ← no warmup (last stage)
```

Activations flow forward between stages via P2P. Gradients flow backward. The 1F1B schedule minimizes memory usage and pipeline bubble time.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `torch not found` | `pip install torch` — or see [pytorch.org](https://pytorch.org/get-started/locally/) for CUDA-specific install |
| `Connection refused` | Check that `RG_PLATFORM_URL` is reachable. The platform must be running. |
| `No tasks available` | Genesis may not be initialized yet. Wait for the network to start a training epoch. |
| `OOM on GPU` | The miner auto-scales batch size for your device. For MPS/CPU it uses batch_size=1, seq_len=512. |
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
