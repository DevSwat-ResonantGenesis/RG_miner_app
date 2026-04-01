#!/usr/bin/env python3
"""
RG MINER APP — Local Mining Client with Platform Auth
======================================================

Standalone app that any user can download to mine on ResonantGenesis:

  1. Login with your platform account (same as Resonant IDE)
  2. View model registry, network status, training data sources
  3. Start real PyTorch training on your GPU
  4. Monitor loss, rewards, and network stats in real-time
  5. Earn $RGT tokens

Usage:
    pip install -r requirements.txt
    python server.py
    # Open http://localhost:3000

Architecture:
    Local FastAPI server → proxies auth + API calls to production
    Same auth flow as Resonant IDE Extension (JWT from platform login)
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rg-miner")

# ── Production endpoints ──
PROD_BASE = os.getenv("RG_PLATFORM_URL", "https://dev-swat.com")
AUTH_URL = f"{PROD_BASE}/api"
MINING_URL = f"{PROD_BASE}/mining"
LIGHTHOUSE_URL = f"{PROD_BASE}/lighthouse"
EXT_CHAIN_URL = f"{PROD_BASE}/ext-chain"
WS_HOST = PROD_BASE.replace("https://", "").replace("http://", "")
MINING_WS_URL = f"wss://{WS_HOST}/ws/mining"

PORT = int(os.getenv("RG_MINER_PORT", "3000"))

# ── Global miner state ──
miner_state = {
    "status": "idle",
    "miner_id": None,
    "jwt_token": None,
    "user_email": None,
    "user_id": None,
    "cycles_completed": 0,
    "cycles_target": 0,
    "total_rgt": 0.0,
    "current_loss": None,
    "loss_history": [],
    "reward_history": [],
    "training_log": [],
    "model_id": "resonant-seed-1b",
    "device": "cpu",
    "real_training": False,
    "start_time": None,
    "error": None,
}

_training_task: Optional[asyncio.Task] = None
_ws_clients: set = set()


# ── Helpers ──

async def broadcast(msg: dict):
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    _ws_clients -= dead


def log_event(msg: str, level: str = "info"):
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "msg": msg}
    miner_state["training_log"].append(entry)
    if len(miner_state["training_log"]) > 500:
        miner_state["training_log"] = miner_state["training_log"][-500:]
    asyncio.create_task(broadcast({"type": "log", **entry}))
    getattr(logger, level if level != "warning" else "warning", logger.info)(msg)


def _detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            return "cuda", True, f"{name} ({mem:.1f}GB)"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", True, "Apple Silicon (MPS)"
        return "cpu", True, "CPU"
    except ImportError:
        return "cpu", False, "CPU (PyTorch not installed)"


# ══════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    dev, torch_ok, dev_name = _detect_device()
    miner_state["device"] = dev
    logger.info(f"RG Miner App starting on http://localhost:{PORT}")
    logger.info(f"  Device: {dev_name}")
    logger.info(f"  PyTorch: {'available' if torch_ok else 'NOT installed'}")
    logger.info(f"  Platform: {PROD_BASE}")
    yield
    if _training_task and not _training_task.done():
        _training_task.cancel()
    logger.info("RG Miner App stopped")


app = FastAPI(title="RG Miner App", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Auth ──

@app.post("/api/login")
async def login(request: Request):
    """Authenticate via platform auth (same flow as Resonant IDE)."""
    body = await request.json()
    email = body.get("email", "")
    password = body.get("password", "")

    if not email or not password:
        return JSONResponse(status_code=400, content={"success": False, "error": "Email and password required"})

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(f"{AUTH_URL}/auth/login", json={"email": email, "password": password})

            if resp.status_code == 200:
                data = resp.json()
                token = data.get("access_token") or data.get("token", "")
                user = data.get("user", {})

                miner_state["jwt_token"] = token
                miner_state["user_email"] = email
                miner_state["user_id"] = user.get("id", "")
                miner_state["miner_id"] = f"miner-{email.split('@')[0]}-{uuid4().hex[:6]}"
                miner_state["error"] = None

                dev, _, dev_name = _detect_device()
                return {
                    "success": True,
                    "email": email,
                    "user_id": miner_state["user_id"],
                    "miner_id": miner_state["miner_id"],
                    "device": dev_name,
                    "platform": PROD_BASE,
                }
            else:
                detail = "Invalid credentials"
                try:
                    detail = resp.json().get("detail", detail)
                except Exception:
                    pass
                return JSONResponse(status_code=401, content={"success": False, "error": detail})
    except httpx.ConnectError:
        return JSONResponse(status_code=503, content={"success": False, "error": f"Cannot reach platform at {PROD_BASE}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/login-token")
async def login_with_token(request: Request):
    """Login with existing JWT (for dev/testing)."""
    body = await request.json()
    token = body.get("token", "")
    email = body.get("email", "user@resonantgenesis.com")

    if not token:
        return JSONResponse(status_code=400, content={"success": False, "error": "Token required"})

    miner_state["jwt_token"] = token
    miner_state["user_email"] = email
    miner_state["miner_id"] = f"miner-{email.split('@')[0]}-{uuid4().hex[:6]}"
    miner_state["error"] = None

    dev, _, dev_name = _detect_device()
    return {"success": True, "email": email, "miner_id": miner_state["miner_id"], "device": dev_name}


@app.post("/api/logout")
async def logout():
    """Clear session."""
    global _training_task
    if _training_task and not _training_task.done():
        _training_task.cancel()
    miner_state.update({"status": "idle", "jwt_token": None, "user_email": None, "miner_id": None, "user_id": None})
    return {"success": True}


# ── Mining controls ──

@app.post("/api/mining/start")
async def start_mining(request: Request):
    global _training_task

    if not miner_state["jwt_token"]:
        raise HTTPException(status_code=401, detail="Login first")
    if miner_state["status"] == "training":
        raise HTTPException(status_code=409, detail="Already mining")

    body = await request.json()
    cycles = body.get("cycles", 5)
    model_id = body.get("model_id", "resonant-seed-1b")
    use_real = body.get("real_training", True)

    miner_state.update({
        "status": "initializing",
        "cycles_target": cycles,
        "cycles_completed": 0,
        "total_rgt": 0.0,
        "loss_history": [],
        "reward_history": [],
        "training_log": [],
        "model_id": model_id,
        "real_training": use_real,
        "start_time": time.time(),
        "error": None,
    })

    _training_task = asyncio.create_task(_mining_loop(cycles, use_real))
    return {"status": "started", "cycles": cycles, "model_id": model_id, "real_training": use_real}


@app.post("/api/mining/stop")
async def stop_mining():
    global _training_task
    if _training_task and not _training_task.done():
        _training_task.cancel()
    miner_state["status"] = "idle"
    log_event("Mining stopped by user", "warning")
    await broadcast({"type": "state", "data": _safe_state()})
    return {"status": "stopped"}


@app.get("/api/mining/state")
async def get_state():
    return _safe_state()


@app.get("/api/system/info")
async def system_info():
    dev, torch_ok, dev_name = _detect_device()
    torch_ver = "not installed"
    if torch_ok:
        import torch
        torch_ver = torch.__version__
    return {
        "device": dev,
        "device_name": dev_name,
        "pytorch_available": torch_ok,
        "pytorch_version": torch_ver,
        "platform_url": PROD_BASE,
        "miner_port": PORT,
    }


# ── Proxy production APIs ──

@app.get("/api/network/health")
async def network_health():
    results = {}
    async with httpx.AsyncClient(timeout=5) as client:
        for name, url in [("mining", f"{MINING_URL}/health"), ("lighthouse", f"{LIGHTHOUSE_URL}/health"), ("blockchain", f"{EXT_CHAIN_URL}/health")]:
            try:
                resp = await client.get(url)
                results[name] = resp.json() if resp.status_code == 200 else {"status": "error", "code": resp.status_code}
            except Exception as e:
                results[name] = {"status": "unreachable", "error": str(e)}
    return results


@app.get("/api/models")
async def get_models():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{MINING_URL}/mining/models")
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/training-data")
async def get_training_data():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{MINING_URL}/mining/training-data")
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/dashboard-data")
async def get_dashboard_data():
    """Proxy the mesh dashboard data."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{MINING_URL}/dashboard/data")
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ── WebSocket for live updates ──

@app.websocket("/ws")
async def dashboard_ws(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        await ws.send_json({"type": "state", "data": _safe_state()})
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("action") == "ping":
                await ws.send_json({"type": "pong"})
            elif msg.get("action") == "get_state":
                await ws.send_json({"type": "state", "data": _safe_state()})
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


# ── Serve HTML ──

@app.get("/", response_class=HTMLResponse)
async def index():
    html = (Path(__file__).parent / "static" / "index.html").read_text()
    return HTMLResponse(content=html)


# ══════════════════════════════════════════════════════════════
# MINING LOOP
# ══════════════════════════════════════════════════════════════

def _safe_state() -> dict:
    """Return miner_state without the JWT token (for frontend)."""
    s = dict(miner_state)
    s.pop("jwt_token", None)
    return s


def _simulate_training(task: dict, cycle: int) -> dict:
    """Simulated training fallback."""
    num_params = 100_000
    k = 10
    indices = sorted(random.sample(range(num_params), k))
    values = [round(random.gauss(0, 0.05), 6) for _ in range(k)]
    hash_input = json.dumps({"indices": indices, "values": values}, sort_keys=True).encode()
    gradient_hash = hashlib.sha256(hash_input).hexdigest()
    base_loss = 11.0 - (cycle * 0.3)
    loss_before = base_loss + random.uniform(-0.1, 0.1)
    loss_after = loss_before - random.uniform(0.5, 1.5)
    return {
        "top_k_indices": indices, "top_k_values": values,
        "original_size": num_params, "compressed_size": k,
        "compression_ratio": num_params / k, "gradient_hash": gradient_hash,
        "loss_before": round(loss_before, 4), "loss_after": round(max(0.5, loss_after), 4),
        "samples_processed": task.get("num_samples", 244140),
        "training_time_seconds": round(random.uniform(20.0, 60.0), 2),
        "real_training": False,
    }


def _real_training_step(task: dict, cycle: int, model, data, device_str: str):
    """Run actual forward/backward pass. Returns grad_data + model/data refs."""
    import torch
    from model_architecture import create_model
    from real_trainer import get_tokenizer, DataShardLoader, RealTrainer, compress_gradients

    device = torch.device(device_str)

    # Init model on first call
    if model is None:
        model_id = task.get("model_id", "resonant-seed-1b")
        model, _ = create_model(model_id)
        model = model.to(device)

    # Init data on first call
    if data is None:
        tok = get_tokenizer()
        loader = DataShardLoader(tok, max_seq_length=task.get("max_seq_length", 4096))
        try:
            data = loader.load_from_huggingface(num_samples=max(200, task.get("batch_size", 8) * 20))
        except Exception:
            data = loader._generate_synthetic_data(max(200, task.get("batch_size", 8) * 20))

    bs = task.get("batch_size", 8)
    lr = task.get("learning_rate", 3e-4)
    start_idx = (cycle * bs) % len(data)
    batch = data[start_idx:start_idx + bs]
    if len(batch) < bs:
        batch = data[:bs]

    input_ids = batch[:, :-1].to(device)
    labels = batch[:, 1:].to(device)

    # Pre-training loss
    model.eval()
    with torch.no_grad():
        pre = model(input_ids=input_ids, labels=labels)
        loss_before = pre["loss"].item()

    # Train
    trainer = RealTrainer(model, task, device)
    result = trainer.train_step(input_ids, labels, lr)
    compressed = compress_gradients(result["gradients"], top_k_ratio=0.01)

    grad_data = {
        "top_k_indices": compressed["top_k_indices"],
        "top_k_values": compressed["top_k_values"],
        "original_size": compressed["original_size"],
        "compressed_size": compressed["compressed_size"],
        "compression_ratio": compressed["compression_ratio"],
        "gradient_hash": compressed["gradient_hash"],
        "loss_before": round(loss_before, 6),
        "loss_after": round(result["loss"], 6),
        "samples_processed": result["samples_in_batch"] * result["seq_length"],
        "training_time_seconds": round(result["training_time_seconds"], 2),
        "device": str(device),
        "grad_norm": result["grad_norm"],
        "real_training": True,
    }
    return grad_data, model, data


async def _mining_loop(num_cycles: int, use_real: bool):
    """Background mining loop — connects to production via WebSocket."""
    try:
        import websockets
    except ImportError:
        log_event("ERROR: pip install websockets", "error")
        miner_state["status"] = "error"
        miner_state["error"] = "websockets not installed"
        return

    token = miner_state["jwt_token"]
    miner_id = miner_state["miner_id"]
    email = miner_state["user_email"]

    dev, torch_ok, dev_name = _detect_device()
    if use_real and not torch_ok:
        use_real = False
        log_event("PyTorch not available — falling back to simulated training", "warning")
    miner_state["real_training"] = use_real
    mode = f"REAL on {dev_name}" if use_real else "SIMULATED"

    log_event(f"Starting mining: {mode} | Model: {miner_state['model_id']} | {num_cycles} cycles")
    await broadcast({"type": "state", "data": _safe_state()})

    try:
        # Phase 1: Lighthouse
        log_event("Phase 1: Registering with Lighthouse...")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{LIGHTHOUSE_URL}/lighthouse/register",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"peer_id": miner_id, "peer_type": "miner", "address": "127.0.0.1", "p2p_port": 8600, "api_port": PORT, "node_version": "1.0.0", "capabilities": ["training", "gradient_submit"]},
                )
                if resp.status_code == 200:
                    peers = resp.json().get("bootstrap_peers", [])
                    log_event(f"Lighthouse: registered — {len(peers)} peers")
                else:
                    log_event(f"Lighthouse: {resp.status_code} — continuing", "warning")
        except Exception as e:
            log_event(f"Lighthouse unavailable: {e}", "warning")

        # Phase 2: Genesis
        log_event("Phase 2: Checking genesis...")
        try:
            async with httpx.AsyncClient(timeout=10, headers={"Authorization": f"Bearer {token}"}) as client:
                resp = await client.get(f"{MINING_URL}/mining/genesis/status")
                if resp.status_code == 200:
                    gs = resp.json()
                    if gs.get("genesis", {}).get("initialized"):
                        log_event("Genesis already initialized")
                    else:
                        log_event("Initializing genesis seed...")
                        await client.post(f"{MINING_URL}/mining/genesis/initialize", json={"model_id": miner_state["model_id"], "miner_ids": [miner_id], "ipfs_base_url": "ipfs://"})
                        log_event("Genesis initialized")
                else:
                    log_event(f"Genesis status: HTTP {resp.status_code}", "warning")
        except Exception as e:
            log_event(f"Genesis check: {e}", "warning")

        # Phase 3: Mining via WebSocket
        miner_state["status"] = "training"
        await broadcast({"type": "state", "data": _safe_state()})
        log_event("Phase 3: Connecting to mining network...")

        ws_url = f"{MINING_WS_URL}?token={token}"

        async with websockets.connect(ws_url, close_timeout=10, ping_interval=30) as ws:
            await ws.send(json.dumps({"action": "register", "miner_id": miner_id, "miner_class": "validator_miner", "account_email": email}))
            welcome = json.loads(await ws.recv())

            if welcome.get("event") != "welcome":
                log_event(f"Registration failed: {welcome}", "error")
                miner_state["status"] = "error"
                miner_state["error"] = str(welcome)
                await broadcast({"type": "state", "data": _safe_state()})
                return

            log_event(f"Connected: {miner_id} | Global step: {welcome.get('param_server', {}).get('global_step', 0)}")

            model = None
            data = None

            for cycle in range(1, num_cycles + 1):
                if miner_state["status"] != "training":
                    break

                log_event(f"━━━ Cycle {cycle}/{num_cycles} ━━━")
                await broadcast({"type": "cycle", "current": cycle, "total": num_cycles})

                # Request task
                await ws.send(json.dumps({"action": "request_task"}))
                task_msg = json.loads(await ws.recv())

                if task_msg.get("event") == "no_tasks":
                    log_event("No tasks available — queue exhausted", "warning")
                    break
                if task_msg.get("event") != "task_assigned":
                    log_event(f"Unexpected response: {task_msg}", "error")
                    break

                task = task_msg["task"]
                log_event(f"Task: {task['task_id'][:12]}... | Epoch {task['epoch']} | Batch {task['batch_index']}")

                # Train
                train_start = time.time()
                if use_real:
                    log_event(f"Training on {dev_name}...")
                    try:
                        grad_data, model, data = await asyncio.to_thread(
                            _real_training_step, task, cycle, model, data, dev,
                        )
                    except Exception as e:
                        log_event(f"Real training failed: {e} — falling back to simulation", "error")
                        grad_data = _simulate_training(task, cycle)
                else:
                    grad_data = _simulate_training(task, cycle)
                    await asyncio.sleep(0.3)

                elapsed = time.time() - train_start
                loss_b = grad_data["loss_before"]
                loss_a = grad_data["loss_after"]
                miner_state["current_loss"] = loss_a
                miner_state["loss_history"].append({"cycle": cycle, "loss_before": loss_b, "loss_after": loss_a, "time": elapsed})

                log_event(f"Loss: {loss_b:.4f} → {loss_a:.4f} (Δ {loss_b - loss_a:.4f}) | {elapsed:.1f}s")
                log_event(f"Compression: {grad_data['original_size']:,} → {grad_data['compressed_size']:,} ({grad_data['compression_ratio']:.0f}x)")
                await broadcast({"type": "training", "cycle": cycle, "loss_before": loss_b, "loss_after": loss_a, "elapsed": elapsed})

                # Submit gradient
                sub_id = f"sub-{uuid4().hex[:8]}"
                await ws.send(json.dumps({
                    "action": "submit_gradient",
                    "gradient": {
                        "submission_id": sub_id, "task_id": task["task_id"],
                        "model_id": task["model_id"], "epoch": task["epoch"],
                        "batch_index": task["batch_index"],
                        "data_shard_hash": task.get("data_shard_hash", ""),
                        "weight_shard_hash": task.get("weight_shard_hash", ""),
                        **grad_data,
                    },
                }))

                submit_msg = json.loads(await ws.recv())
                if submit_msg.get("event") == "gradient_accepted":
                    reward = submit_msg.get("reward", 0)
                    miner_state["total_rgt"] += reward
                    miner_state["cycles_completed"] = cycle
                    miner_state["reward_history"].append({"cycle": cycle, "reward": reward})
                    log_event(f"✓ Gradient ACCEPTED — Reward: {reward} $RGT | Total: {miner_state['total_rgt']:.2f} $RGT")

                    try:
                        agg_raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        agg = json.loads(agg_raw)
                        if agg.get("event") == "aggregation_complete":
                            log_event(f"⚡ Aggregation — global step {agg['global_step']} ({agg['layers_merged']} layers)")
                    except asyncio.TimeoutError:
                        pass

                elif submit_msg.get("event") == "gradient_rejected":
                    log_event(f"✗ Gradient REJECTED: {submit_msg.get('reason')}", "error")
                else:
                    log_event(f"Unexpected: {submit_msg}", "error")

                await broadcast({"type": "state", "data": _safe_state()})

                # Heartbeat
                await ws.send(json.dumps({"action": "heartbeat"}))
                await ws.recv()

            # Done
            log_event(f"Mining complete: {miner_state['cycles_completed']}/{num_cycles} cycles | {miner_state['total_rgt']:.2f} $RGT earned")
            miner_state["status"] = "idle"
            await broadcast({"type": "state", "data": _safe_state()})
            await broadcast({"type": "complete", "cycles": miner_state["cycles_completed"], "rgt": miner_state["total_rgt"]})

    except asyncio.CancelledError:
        log_event("Mining cancelled", "warning")
        miner_state["status"] = "idle"
    except Exception as e:
        log_event(f"Mining error: {e}", "error")
        miner_state["status"] = "error"
        miner_state["error"] = str(e)
        await broadcast({"type": "state", "data": _safe_state()})


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print()
    print("=" * 60)
    print("  RG MINER APP — ResonantGenesis Network")
    print("=" * 60)
    print(f"  Dashboard: http://localhost:{PORT}")
    print(f"  Platform:  {PROD_BASE}")
    dev, tok, dname = _detect_device()
    print(f"  Device:    {dname}")
    print("=" * 60)
    print()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
