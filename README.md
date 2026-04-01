# RG Miner App ⛏️

**Standalone mining client for the ResonantGenesis decentralized LLM training network.**

Download this app, login with your platform account, and start earning $RGT tokens by training AI models on your GPU.

## Quick Start

```bash
# Clone
git clone https://github.com/DevSwat-ResonantGenesis/RG_miner_app.git
cd RG_miner_app

# Install dependencies
pip install -r requirements.txt

# Run
python server.py

# Open http://localhost:3000 in your browser
```

## Features

- **Platform Login** — Authenticate with your ResonantGenesis account (same as Resonant IDE)
- **Real GPU Training** — Runs actual PyTorch forward/backward passes on your CUDA/MPS/CPU
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
| PyTorch | 2.1+ | 2.4+ |
| Network | Internet connection to dev-swat.com | — |

## How It Works

1. **Login** → Your credentials are sent to the platform auth service (same as the web app and IDE extension)
2. **Start Mining** → The app connects to the Mining service via WebSocket
3. **Receive Task** → You get a training task (model weights shard + data shard)
4. **Train** → Real forward/backward pass on your GPU (or simulated for testing)
5. **Submit Gradient** → Compressed gradient (Top-K 1%) with SHA256 hash
6. **Earn Reward** → $RGT tokens credited to your wallet

## Configuration

Set environment variables or use defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `RG_PLATFORM_URL` | `https://dev-swat.com` | Platform base URL |
| `RG_MINER_PORT` | `3000` | Local dashboard port |

## Architecture

```
RG_miner_app/
├── server.py          # FastAPI backend — auth proxy, mining loop, WebSocket
├── static/
│   └── index.html     # Single-page dashboard (vanilla JS, no build step)
├── model_architecture.py  # PyTorch Transformer (GQA + RoPE + SwiGLU)
├── real_trainer.py    # Training engine, tokenizer, gradient compression
├── requirements.txt
├── Dockerfile
└── README.md
```

## Training Modes

- **Real Training** (default) — Requires PyTorch. Runs actual model forward/backward on GPU/CPU.
- **Simulated** — No PyTorch needed. Generates valid gradient structures for testing.

Toggle in the dashboard UI or pass `real_training: false` when starting.

## License

Part of the ResonantGenesis platform. See [dev-swat.com](https://dev-swat.com) for details.
