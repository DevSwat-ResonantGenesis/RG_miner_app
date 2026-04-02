#!/usr/bin/env python3
"""
RG MINER APP — Local Mining Client with Platform Auth
======================================================

Standalone app that any user can download to mine on ResonantGenesis:

  1. Click "Sign in with ResonantGenesis" → opens platform login in browser
  2. Platform authenticates → redirects token back to local app
  3. Start real PyTorch training on your GPU
  4. Monitor loss, rewards, and network stats in real-time
  5. Earn $RGT tokens

Auth flow (identical to Resonant IDE Extension):
  1. App opens browser → https://dev-swat.com/auth/desktop-callback?port=3000
  2. Platform checks if logged in → if not, shows login page
  3. After login, platform redirects → http://127.0.0.1:3000/auth-callback?token=JWT
  4. Local app stores token, shows dashboard

Usage:
    pip install -r requirements.txt
    python server.py
    # Open http://localhost:3000
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import platform
import random
import secrets
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rg-miner")

# ── Production endpoints — dual-domain fallback (same as Resonant IDE) ──
PLATFORM_DOMAINS = [
    os.getenv("RG_PLATFORM_URL", "https://dev-swat.com"),
    "https://resonantgenesis.xyz",
]
PROD_BASE = PLATFORM_DOMAINS[0]  # Updated at startup after health check

PORT = int(os.getenv("RG_MINER_PORT", "3000"))

# ── Encrypted token storage ──
_TOKEN_DIR = Path.home() / ".rg_miner"
_TOKEN_FILE = _TOKEN_DIR / "auth.enc"


def _get_fernet() -> Fernet:
    """Derive a machine-specific encryption key (same concept as OS keyring)."""
    salt = (platform.node() + "-rg-miner-salt").encode()
    machine_id = (platform.node() + platform.machine() + os.getenv("USER", "default")).encode()
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480_000)
    key = base64.urlsafe_b64encode(kdf.derive(machine_id))
    return Fernet(key)


def _save_token_encrypted(token: str, email: str, user_id: str, user_name: str):
    """Persist auth to encrypted file on disk."""
    try:
        _TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"jwt": token, "email": email, "user_id": user_id, "user_name": user_name, "ts": time.time()})
        _TOKEN_FILE.write_bytes(_get_fernet().encrypt(payload.encode()))
        _TOKEN_FILE.chmod(0o600)
    except Exception as e:
        logger.warning(f"Failed to save encrypted token: {e}")


def _load_token_encrypted() -> Optional[dict]:
    """Load auth from encrypted file. Returns None if missing/invalid."""
    try:
        if not _TOKEN_FILE.exists():
            return None
        data = json.loads(_get_fernet().decrypt(_TOKEN_FILE.read_bytes()).decode())
        # Check JWT expiry
        if _is_jwt_expired(data.get("jwt", "")):
            logger.info("Stored token expired — clearing")
            _clear_token_file()
            return None
        return data
    except Exception as e:
        logger.warning(f"Failed to load encrypted token: {e}")
        return None


def _clear_token_file():
    """Remove encrypted token file."""
    try:
        if _TOKEN_FILE.exists():
            _TOKEN_FILE.unlink()
    except Exception:
        pass


def _is_jwt_expired(token: str) -> bool:
    """Decode JWT payload and check exp claim (no signature verification — server does that)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return True
        payload = parts[1]
        # Add padding
        payload += "=" * (4 - len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        exp = data.get("exp", 0)
        return time.time() > exp
    except Exception:
        return True


async def _health_check_domain(domain: str) -> bool:
    """Quick health check — 3s timeout (same as Resonant IDE)."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.head(f"{domain}/api/v1/health")
            return resp.status_code < 500
    except Exception:
        return False


async def _resolve_platform() -> str:
    """Find first reachable domain (dual-domain fallback like Resonant IDE)."""
    global PROD_BASE
    for domain in PLATFORM_DOMAINS:
        if await _health_check_domain(domain):
            PROD_BASE = domain
            return domain
    PROD_BASE = PLATFORM_DOMAINS[0]
    return PROD_BASE


_CHECKPOINT_DIR = _TOKEN_DIR / "checkpoints"


def _save_checkpoint(model, model_id: str, step: int) -> Optional[str]:
    """Save model checkpoint locally. Returns path or None."""
    try:
        import torch
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{model_id}_step{step}.pt"
        path = _CHECKPOINT_DIR / filename
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_id": model_id,
            "step": step,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, str(path))
        # Keep only latest 3 checkpoints per model
        existing = sorted(_CHECKPOINT_DIR.glob(f"{model_id}_step*.pt"), key=lambda p: p.stat().st_mtime)
        for old in existing[:-3]:
            old.unlink()
        logger.info(f"Checkpoint saved: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
        return str(path)
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")
        return None


def _load_latest_checkpoint(model_id: str = "resonant-seed-1b"):
    """Load latest checkpoint if exists. Returns (model, step) or (None, 0)."""
    try:
        import torch
        from model_architecture import create_model
        if not _CHECKPOINT_DIR.exists():
            return None, 0
        checkpoints = sorted(_CHECKPOINT_DIR.glob(f"{model_id}_step*.pt"), key=lambda p: p.stat().st_mtime)
        if not checkpoints:
            return None, 0
        latest = checkpoints[-1]
        data = torch.load(str(latest), map_location="cpu")
        model, _ = create_model(model_id)
        model.load_state_dict(data["model_state_dict"])
        step = data.get("step", 0)
        logger.info(f"Loaded checkpoint: {latest.name} (step {step}, {latest.stat().st_size / 1024 / 1024:.1f} MB)")
        return model, step
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None, 0


def _list_checkpoints() -> list:
    """List all local checkpoints with metadata."""
    result = []
    if not _CHECKPOINT_DIR.exists():
        return result
    for path in sorted(_CHECKPOINT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            import torch
            data = torch.load(str(path), map_location="cpu", weights_only=False)
            result.append({
                "filename": path.name,
                "model_id": data.get("model_id", "unknown"),
                "step": data.get("step", 0),
                "saved_at": data.get("saved_at", ""),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
                "path": str(path),
            })
        except Exception:
            result.append({"filename": path.name, "size_mb": round(path.stat().st_size / 1024 / 1024, 1), "path": str(path)})
    return result


def _platform_urls():
    """Return current platform URLs derived from PROD_BASE."""
    host = PROD_BASE.replace("https://", "").replace("http://", "")
    return {
        "auth": f"{PROD_BASE}/api",
        "mining": f"{PROD_BASE}/mining",
        "lighthouse": f"{PROD_BASE}/lighthouse",
        "ext_chain": f"{PROD_BASE}/ext-chain",
        "ws_mining": f"wss://{host}/ws/mining",
    }


async def _verify_token_with_platform(token: str) -> Optional[dict]:
    """Re-verify JWT with platform /auth/me — returns user info or None."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{PROD_BASE}/auth/me",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None

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
    "detailed_metrics": [],  # Per-cycle: epoch, grad_norm, throughput, LR, compression, etc.
    "model_id": "resonant-seed-1b",
    "device": "cpu",
    "real_training": False,
    "start_time": None,
    "error": None,
    # ── Shard state ──
    "shard_assignment": None,   # ShardAssignment dict from server (layer_start, layer_end, etc.)
    "pipeline_group_id": None,
    "stage_index": None,
    "is_sharded": False,
    "upstream_peer": None,      # Miner ID of upstream peer (sends activations to us)
    "downstream_peer": None,    # Miner ID of downstream peer (we send activations to)
}

_training_task: Optional[asyncio.Task] = None
_ws_clients: set = set()
_session_token: Optional[str] = None  # Random token set on auth, checked via cookie
_csrf_token: Optional[str] = None  # CSRF token, regenerated per session
_model_ref = None  # Persistent model reference for download
_activation_buffer = {}  # transfer_id → {chunks: [...], metadata: {...}} for P2P receive
_peer_addresses = {}  # miner_id → "http://host:port" for P2P activation routing


# ── Helpers ──

async def broadcast(msg: dict):
    global _ws_clients
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


def _check_csrf(request: Request):
    """Validate CSRF token on state-changing requests."""
    if not _csrf_token:
        return  # No session yet — skip
    header_token = request.headers.get("x-csrf-token", "")
    if header_token != _csrf_token:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")


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
# SHARD HELPERS — Pipeline-parallel model loading
# ══════════════════════════════════════════════════════════════

def _get_gpu_capability() -> dict:
    """Detect GPU hardware for shard capability registration."""
    try:
        import torch
        import psutil
    except ImportError:
        psutil = None

    cap = {
        "gpu_model": "",
        "gpu_vram_gb": 0.0,
        "system_ram_gb": 0.0,
        "cpu_cores": os.cpu_count() or 1,
        "bandwidth_mbps": 1000.0,  # Default — real measurement TBD
        "storage_available_gb": 0.0,
        "location_region": "unknown",
        "supported_dtypes": ["fp32"],
    }

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cap["gpu_model"] = props.name
            cap["gpu_vram_gb"] = round(props.total_mem / 1e9, 1)
            cap["supported_dtypes"] = ["fp32", "fp16", "bf16"]
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cap["gpu_model"] = "Apple Silicon (MPS)"
            # MPS doesn't expose VRAM — estimate from system RAM
            cap["gpu_vram_gb"] = 8.0  # Conservative
            cap["supported_dtypes"] = ["fp32", "fp16"]
    except Exception:
        pass

    if psutil:
        try:
            cap["system_ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
            cap["storage_available_gb"] = round(psutil.disk_usage("/").free / 1e9, 1)
        except Exception:
            pass

    return cap


async def _register_capability(miner_id: str, token: str) -> bool:
    """Register this miner's hardware capabilities with the mining server."""
    urls = _platform_urls()
    cap = _get_gpu_capability()
    payload = {"miner_id": miner_id, **cap}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{urls['mining']}/mining/shards/register-capability",
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
            )
            if resp.status_code == 200:
                log_event(f"Shard capability registered: {cap['gpu_model']} ({cap['gpu_vram_gb']} GB VRAM)")
                return True
            else:
                log_event(f"Shard capability registration: HTTP {resp.status_code}", "warning")
    except Exception as e:
        log_event(f"Shard capability registration failed: {e}", "warning")
    return False


async def _fetch_shard_assignment(miner_id: str, token: str) -> Optional[dict]:
    """Fetch this miner's shard assignment from the mining server."""
    urls = _platform_urls()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{urls['mining']}/mining/shards/assignment/{miner_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code == 200:
                assignment = resp.json()
                log_event(
                    f"Shard assigned: stage {assignment['stage_index']}/{assignment['num_stages']} | "
                    f"layers {assignment['layer_start']}-{assignment['layer_end']} | "
                    f"group {assignment['pipeline_group_id'][:12]}..."
                )
                return assignment
            elif resp.status_code == 404:
                return None  # No assignment yet — full model training
            else:
                log_event(f"Shard assignment check: HTTP {resp.status_code}", "warning")
    except Exception as e:
        log_event(f"Shard assignment check failed: {e}", "warning")
    return None


async def _fetch_pipeline_peers(miner_id: str, token: str) -> dict:
    """Fetch upstream/downstream peers for P2P activation routing."""
    urls = _platform_urls()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{urls['mining']}/mining/shards/pipeline-peers/{miner_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return {"peers": [], "upstream": None, "downstream": None}


async def _report_shard_ready(miner_id: str, token: str) -> bool:
    """Tell the server this miner has loaded its shard and is ready."""
    urls = _platform_urls()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{urls['mining']}/mining/shards/report-ready/{miner_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            return resp.status_code == 200
    except Exception:
        return False


async def _report_shard_loaded(miner_id: str, token: str, assignment: dict) -> bool:
    """Report to the server that we've loaded our shard weights — registers us as a P2P source."""
    urls = _platform_urls()
    payload = {
        "miner_id": miner_id,
        "model_id": assignment.get("model_id", miner_state.get("model_id", "resonant-seed-1b")),
        "layer_start": assignment["layer_start"],
        "layer_end": assignment["layer_end"],
        "weight_hash": miner_state.get("_shard_weight_hash", ""),
        "size_bytes": miner_state.get("_shard_size_bytes", 0),
        "num_params": miner_state.get("_shard_num_params", 0),
        "miner_address": f"127.0.0.1:{PORT}",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{urls['mining']}/mining/weights/report-loaded",
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
            )
            if resp.status_code == 200:
                data = resp.json()
                log_event(
                    f"Weight registry: shard registered "
                    f"(key={data.get('shard_key', '?')}, "
                    f"replicas={data.get('replicas', '?')})"
                )
                return True
            else:
                log_event(f"Weight registry report: HTTP {resp.status_code}", "warning")
    except Exception as e:
        log_event(f"Weight registry report failed: {e}", "warning")
    return False


async def _request_weight_transfer_plan(miner_id: str, token: str, assignment: dict) -> Optional[dict]:
    """
    Request a weight transfer plan from the server.
    
    The server tells us which peers have our assigned layers loaded,
    so we can pull weights from them via P2P instead of initializing from scratch.
    """
    urls = _platform_urls()
    payload = {
        "miner_id": miner_id,
        "model_id": assignment.get("model_id", miner_state.get("model_id", "resonant-seed-1b")),
        "layer_start": assignment["layer_start"],
        "layer_end": assignment["layer_end"],
        "include_embedding": assignment.get("has_embedding", False),
        "include_lm_head": assignment.get("has_lm_head", False),
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{urls['mining']}/mining/weights/request-transfer",
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
            )
            if resp.status_code == 200:
                plan = resp.json()
                peer_sources = [s for s in plan.get("sources", []) if s.get("type") == "peer"]
                log_event(
                    f"Weight transfer plan: {len(peer_sources)} peer sources, "
                    f"{plan.get('total_mb', 0)} MB total"
                )
                return plan
    except Exception as e:
        log_event(f"Weight transfer plan request failed: {e}", "warning")
    return None


async def _download_weights_from_peer(peer_address: str, model_id: str, layer_start: int, layer_end: int) -> Optional[dict]:
    """
    Download weight tensors from a peer miner via P2P.
    
    Returns the state_dict if successful, None otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(
                f"http://{peer_address}/p2p/serve-weights",
                params={
                    "model_id": model_id,
                    "layer_start": layer_start,
                    "layer_end": layer_end,
                },
            )
            if resp.status_code == 200:
                log_event(f"P2P: Downloaded weights from {peer_address}")
                return resp.json()
    except Exception as e:
        log_event(f"P2P weight download from {peer_address} failed: {e}", "warning")
    return None


def _create_model_or_shard(task: dict, assignment: Optional[dict], device_str: str):
    """
    Create either a full model or a ModelShard depending on assignment.
    
    If assignment is None: creates full model (original behavior).
    If assignment is set: creates only our pipeline shard (partial model).
    """
    import torch
    import hashlib

    if assignment is None:
        # Original full-model path
        from model_architecture import create_model
        model_id = task.get("model_id", "resonant-seed-1b")
        model, _ = create_model(model_id)
        return model.to(torch.device(device_str))

    # Shard path — only load our layers
    from moe_architecture import create_model_shard
    model_id = task.get("model_id", assignment.get("model_id", "resonant-seed-1b"))
    layer_start = assignment["layer_start"]
    layer_end = assignment["layer_end"]
    has_embedding = assignment.get("has_embedding", False)
    has_lm_head = assignment.get("has_lm_head", False)

    shard, config = create_model_shard(
        model_id=model_id,
        layer_start=layer_start,
        layer_end=layer_end,
        has_embedding=has_embedding,
        has_lm_head=has_lm_head,
    )

    device = torch.device(device_str)
    shard = shard.to(device)

    num_params = sum(p.numel() for p in shard.parameters())
    size_bytes = num_params * 2  # fp16

    # Compute weight hash for integrity verification
    h = hashlib.sha256()
    for name, param in sorted(shard.named_parameters()):
        h.update(name.encode())
        h.update(param.detach().cpu().numpy().tobytes())
    weight_hash = h.hexdigest()

    # Store shard metadata for reporting to the registry
    miner_state["_shard_weight_hash"] = weight_hash
    miner_state["_shard_num_params"] = num_params
    miner_state["_shard_size_bytes"] = size_bytes

    logger.info(
        f"ModelShard loaded: layers {layer_start}-{layer_end} | "
        f"{num_params:,} params ({size_bytes / 1e6:.0f} MB) | "
        f"embed={has_embedding} lm_head={has_lm_head} | "
        f"hash={weight_hash[:16]}..."
    )
    return shard


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
    # Dual-domain health check (same as Resonant IDE)
    resolved = await _resolve_platform()
    logger.info(f"  Platform: {resolved}")
    # Try to restore session from encrypted token file
    stored = _load_token_encrypted()
    if stored and stored.get("jwt"):
        logger.info(f"  Restored session: {stored.get('email', '?')} (encrypted storage)")
        miner_state["jwt_token"] = stored["jwt"]
        miner_state["user_email"] = stored.get("email", "")
        miner_state["user_id"] = stored.get("user_id", "")
        miner_state["user_name"] = stored.get("user_name", "")
        miner_state["miner_id"] = f"miner-{stored.get('email', 'user').split('@')[0]}-{uuid4().hex[:6]}"
    # Try to load latest checkpoint
    global _model_ref
    model, step = _load_latest_checkpoint(miner_state["model_id"])
    if model is not None:
        _model_ref = model
        miner_state["cycles_completed"] = step
        logger.info(f"  Model restored: {miner_state['model_id']} at step {step}")
    yield
    if _training_task and not _training_task.done():
        _training_task.cancel()
    logger.info("RG Miner App stopped")


app = FastAPI(title="RG Miner App", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ══════════════════════════════════════════════════════════════
# AUTH — Desktop Callback (same as Resonant IDE Extension)
# ══════════════════════════════════════════════════════════════

@app.get("/auth-callback")
async def auth_callback(request: Request):
    """
    Receives JWT token from platform after browser login.
    Flow: platform redirects → http://127.0.0.1:3000/auth-callback?token=JWT
    This is identical to how the Resonant IDE Extension receives tokens.
    """
    token = request.query_params.get("token", "")
    if not token:
        return HTMLResponse(content="<h2>Authentication failed</h2><p>No token received. <a href='/'>Try again</a></p>", status_code=400)

    # Check JWT expiry before proceeding
    if _is_jwt_expired(token):
        return HTMLResponse(content="<h2>Token expired</h2><p>Please log in again. <a href='/'>Back</a></p>", status_code=401)

    # Verify with platform — same as Resonant IDE (GET /auth/me)
    user_info = await _verify_token_with_platform(token)
    if user_info:
        email = user_info.get("email", "")
        user_id = user_info.get("user_id", user_info.get("id", ""))
        user_name = user_info.get("full_name", user_info.get("name", ""))
        logger.info(f"Verified user: {email} ({user_name})")
    else:
        logger.warning("Platform verification failed — rejecting token")
        return HTMLResponse(content="<h2>Verification failed</h2><p>Could not verify with platform. <a href='/'>Try again</a></p>", status_code=401)

    global _session_token, _csrf_token
    _session_token = secrets.token_urlsafe(32)
    _csrf_token = secrets.token_urlsafe(32)

    miner_state["jwt_token"] = token
    miner_state["user_email"] = email
    miner_state["user_id"] = user_id
    miner_state["user_name"] = user_name
    miner_state["miner_id"] = f"miner-{email.split('@')[0]}-{uuid4().hex[:6]}"
    miner_state["error"] = None

    # Persist to encrypted file (survives restarts)
    _save_token_encrypted(token, email, user_id, user_name)

    dev, _, dev_name = _detect_device()
    logger.info(f"Authenticated: {email} | Device: {dev_name}")

    # Show success page that auto-redirects to dashboard (set session cookie)
    response = HTMLResponse(content=f"""
    <!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>RG Miner — Authenticated</title>
    <style>body{{background:#0a0a0f;color:#e0e0e8;font-family:system-ui;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}}
    .card{{background:#12121a;border:1px solid #2a2a3e;border-radius:16px;padding:40px;text-align:center;max-width:400px}}
    .check{{font-size:48px;margin-bottom:16px}} .email{{color:#a29bfe;font-weight:600}} .btn{{display:inline-block;margin-top:20px;padding:12px 32px;background:#6c5ce7;color:white;border-radius:8px;text-decoration:none;font-weight:600}}</style>
    <script>setTimeout(()=>window.location.href='/',1500)</script>
    </head><body><div class="card">
    <div class="check">✓</div>
    <h2>Authenticated!</h2>
    <p>Signed in as <span class="email">{email}</span></p>
    <p style="color:#8888a0;font-size:14px">Redirecting to dashboard...</p>
    <a class="btn" href="/">Open Dashboard</a>
    </div></body></html>
    """)
    response.set_cookie(
        key="rg_session",
        value=_session_token,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=86400,  # 24 hours
    )
    return response


@app.get("/api/auth/url")
async def get_auth_url():
    """Returns the platform auth URL for the browser redirect."""
    auth_url = f"{PROD_BASE}/auth/desktop-callback?port={PORT}"
    return {"auth_url": auth_url, "platform": PROD_BASE, "port": PORT}


@app.get("/api/auth/status")
async def auth_status(request: Request):
    """Check if user is authenticated — requires valid session cookie."""
    global _session_token, _csrf_token
    cookie = request.cookies.get("rg_session")

    # Case 1: Valid session cookie
    if _session_token and cookie == _session_token and miner_state["jwt_token"]:
        # Check JWT not expired
        if _is_jwt_expired(miner_state["jwt_token"]):
            _session_token = None
            _csrf_token = None
            _clear_token_file()
            miner_state.update({"jwt_token": None, "user_email": None, "miner_id": None})
            return {"authenticated": False, "reason": "token_expired"}
        return {"authenticated": True, "email": miner_state["user_email"], "miner_id": miner_state["miner_id"], "csrf_token": _csrf_token}

    # Case 2: No cookie but server has restored session from encrypted file — issue new session
    if not _session_token and miner_state["jwt_token"] and not _is_jwt_expired(miner_state["jwt_token"]):
        _session_token = secrets.token_urlsafe(32)
        _csrf_token = secrets.token_urlsafe(32)
        return {"authenticated": True, "email": miner_state["user_email"], "miner_id": miner_state["miner_id"], "csrf_token": _csrf_token, "set_session": _session_token}

    return {"authenticated": False}


@app.post("/api/logout")
async def logout(request: Request):
    """Clear session, cookie, CSRF, and encrypted token file."""
    global _training_task, _session_token, _csrf_token
    # CSRF check on logout too
    _check_csrf(request)
    if _training_task and not _training_task.done():
        _training_task.cancel()
    _session_token = None
    _csrf_token = None
    _clear_token_file()
    miner_state.update({"status": "idle", "jwt_token": None, "user_email": None, "miner_id": None, "user_id": None})
    response = JSONResponse({"success": True})
    response.delete_cookie("rg_session", path="/")
    return response


# ── Mining controls ──

@app.post("/api/mining/start")
async def start_mining(request: Request):
    global _training_task

    _check_csrf(request)
    if not miner_state["jwt_token"]:
        raise HTTPException(status_code=401, detail="Login first")
    # Re-verify JWT with platform before starting mining
    if _is_jwt_expired(miner_state["jwt_token"]):
        raise HTTPException(status_code=401, detail="Token expired — please re-login")
    user_info = await _verify_token_with_platform(miner_state["jwt_token"])
    if not user_info:
        raise HTTPException(status_code=401, detail="Token invalid — please re-login")
    if miner_state["status"] == "training":
        raise HTTPException(status_code=409, detail="Already mining")

    body = await request.json()
    cycles = body.get("cycles", 5)
    model_id = body.get("model_id", "resonant-seed-1b")

    miner_state.update({
        "status": "initializing",
        "cycles_target": cycles,
        "cycles_completed": 0,
        "total_rgt": 0.0,
        "loss_history": [],
        "reward_history": [],
        "training_log": [],
        "model_id": model_id,
        "real_training": True,
        "start_time": time.time(),
        "error": None,
    })

    _training_task = asyncio.create_task(_mining_loop(cycles))
    return {"status": "started", "cycles": cycles, "model_id": model_id}


@app.post("/api/mining/stop")
async def stop_mining(request: Request):
    global _training_task
    _check_csrf(request)
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
    urls = _platform_urls()
    results = {}
    async with httpx.AsyncClient(timeout=5) as client:
        for name, url in [("mining", f"{urls['mining']}/health"), ("lighthouse", f"{urls['lighthouse']}/health"), ("blockchain", f"{urls['ext_chain']}/health")]:
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
            resp = await client.get(f"{_platform_urls()['mining']}/mining/models")
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/training-data")
async def get_training_data():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{_platform_urls()['mining']}/mining/training-data")
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/dashboard-data")
async def get_dashboard_data():
    """Proxy the mesh dashboard data."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{_platform_urls()['mining']}/dashboard/data")
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ── Detailed Metrics ──

@app.get("/api/metrics")
async def get_metrics():
    """Return detailed per-cycle training metrics."""
    return {
        "metrics": miner_state["detailed_metrics"],
        "summary": _compute_metrics_summary(),
    }


@app.get("/api/param-server")
async def get_param_server():
    """Proxy parameter server stats + miner list from mining service."""
    urls = _platform_urls()
    token = miner_state.get("jwt_token", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    result = {}
    try:
        async with httpx.AsyncClient(timeout=10, headers=headers) as client:
            for key, path in [("stats", "/mining/param-server/stats"), ("miners", "/mining/miners"), ("tasks", "/mining/tasks/stats")]:
                try:
                    resp = await client.get(f"{urls['mining']}{path}")
                    result[key] = resp.json() if resp.status_code == 200 else {"error": resp.status_code}
                except Exception as e:
                    result[key] = {"error": str(e)}
    except Exception as e:
        result["error"] = str(e)
    return result


def _compute_metrics_summary() -> dict:
    """Compute aggregate summary from detailed metrics."""
    metrics = miner_state["detailed_metrics"]
    if not metrics:
        return {}
    total_time = sum(m.get("training_time", 0) for m in metrics)
    total_samples = sum(m.get("samples_processed", 0) for m in metrics)
    losses = [m["loss_after"] for m in metrics if "loss_after" in m]
    grad_norms = [m["grad_norm"] for m in metrics if "grad_norm" in m]
    return {
        "total_cycles": len(metrics),
        "total_training_time": round(total_time, 1),
        "total_samples": total_samples,
        "avg_loss": round(sum(losses) / len(losses), 6) if losses else None,
        "min_loss": round(min(losses), 6) if losses else None,
        "max_loss": round(max(losses), 6) if losses else None,
        "avg_grad_norm": round(sum(grad_norms) / len(grad_norms), 4) if grad_norms else None,
        "throughput_samples_per_sec": round(total_samples / total_time, 1) if total_time > 0 else 0,
    }


# ── Model Download ──

@app.get("/api/model/info")
async def model_info():
    """Get info about the locally trained model + saved checkpoints."""
    has_model = _model_ref is not None
    checkpoints = _list_checkpoints()
    return {
        "in_memory": has_model,
        "model_id": miner_state["model_id"],
        "cycles_trained": miner_state["cycles_completed"],
        "device": miner_state["device"],
        "checkpoints": checkpoints,
        "checkpoint_dir": str(_CHECKPOINT_DIR),
    }


@app.get("/api/model/download")
async def model_download(request: Request, checkpoint: Optional[str] = None):
    """Download model weights as a .pt file. Optionally specify a checkpoint filename."""
    cookie = request.cookies.get("rg_session")
    if not (_session_token and cookie == _session_token):
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        import torch
        import io
        from fastapi.responses import StreamingResponse

        if checkpoint:
            # Download a specific checkpoint from disk
            ckpt_path = _CHECKPOINT_DIR / checkpoint
            if not ckpt_path.exists() or not str(ckpt_path).startswith(str(_CHECKPOINT_DIR)):
                raise HTTPException(status_code=404, detail="Checkpoint not found")
            return StreamingResponse(
                open(str(ckpt_path), "rb"),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f'attachment; filename="{checkpoint}"'},
            )

        if _model_ref is None:
            # Try loading latest checkpoint from disk
            checkpoints = _list_checkpoints()
            if checkpoints:
                ckpt_path = checkpoints[0]["path"]
                return StreamingResponse(
                    open(ckpt_path, "rb"),
                    media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{Path(ckpt_path).name}"'},
                )
            raise HTTPException(status_code=404, detail="No model in memory and no checkpoints on disk — train at least 1 cycle first")

        buffer = io.BytesIO()
        torch.save(_model_ref.state_dict(), buffer)
        buffer.seek(0)
        model_id = miner_state["model_id"]
        step = miner_state["cycles_completed"]
        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{model_id}_step{step}.pt"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export model: {e}")


@app.get("/api/model/load-checkpoint")
async def load_checkpoint_endpoint(request: Request):
    """Load latest checkpoint from disk into memory (after restart)."""
    cookie = request.cookies.get("rg_session")
    if not (_session_token and cookie == _session_token):
        raise HTTPException(status_code=401, detail="Authentication required")
    global _model_ref
    model, step = _load_latest_checkpoint(miner_state["model_id"])
    if model is not None:
        _model_ref = model
        miner_state["cycles_completed"] = step
        return {"status": "loaded", "model_id": miner_state["model_id"], "step": step}
    return {"status": "no_checkpoint", "message": "No checkpoint files found"}


@app.get("/api/model/network-status")
async def network_model_status():
    """Query the parameter server for the current global model state."""
    urls = _platform_urls()
    token = miner_state.get("jwt_token", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    result = {"global_step": None, "active_miners": None, "total_samples": None, "weight_shards": []}
    try:
        async with httpx.AsyncClient(timeout=10, headers=headers) as client:
            # Param server stats
            resp = await client.get(f"{urls['mining']}/mining/param-server/stats")
            if resp.status_code == 200:
                stats = resp.json()
                result["global_step"] = stats.get("global_step", 0)
                result["active_miners"] = stats.get("active_miners", 0)
                result["total_samples"] = stats.get("total_samples_trained", 0)
                result["total_gradients"] = stats.get("total_gradients_received", 0)
                result["aggregation_rounds"] = stats.get("total_aggregation_rounds", 0)
            # Genesis status for model info
            resp2 = await client.get(f"{urls['mining']}/mining/genesis/status")
            if resp2.status_code == 200:
                gs = resp2.json()
                genesis = gs.get("genesis", gs)
                result["model_id"] = genesis.get("model_id") or genesis.get("model_config", {}).get("model_id")
                result["initialized"] = genesis.get("initialized", False)
    except Exception as e:
        result["error"] = str(e)
    return result


# ── LLM Inference Proxy ──

@app.post("/api/inference/chat")
async def inference_chat(request: Request):
    """Proxy chat completions to platform LLM providers."""
    cookie = request.cookies.get("rg_session")
    if not (_session_token and cookie == _session_token):
        raise HTTPException(status_code=401, detail="Authentication required")
    _check_csrf(request)
    token = miner_state.get("jwt_token", "")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated with platform")
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{PROD_BASE}/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=body,
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {e}")


@app.get("/api/inference/models")
async def inference_models():
    """List available LLM models from platform."""
    token = miner_state.get("jwt_token", "")
    if not token:
        return {"models": [], "error": "Not authenticated"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{PROD_BASE}/api/v1/models",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code == 200:
                return resp.json()
            return {"models": [], "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"models": [], "error": str(e)}


# ══════════════════════════════════════════════════════════════
# P2P ACTIVATION TRANSPORT — Pipeline-parallel sidecar
# ══════════════════════════════════════════════════════════════

_pending_activations = asyncio.Queue(maxsize=16)  # Incoming activations from upstream
_pending_gradients_back = asyncio.Queue(maxsize=16)  # Incoming gradients from downstream


@app.post("/p2p/receive-activation")
async def receive_activation(request: Request):
    """
    Receive activation tensor from upstream pipeline stage.
    
    Forward pass: Miner N finishes → compresses → POST here on Miner N+1.
    The training loop picks this up from _pending_activations queue.
    """
    try:
        body = await request.json()
        transfer_id = body.get("transfer_id", "unknown")
        direction = body.get("direction", "forward")

        if direction == "forward":
            await asyncio.wait_for(
                _pending_activations.put(body),
                timeout=30.0,
            )
            log_event(f"P2P: received activation {transfer_id[:12]}... from upstream ({body.get('source_miner', '?')})")
        elif direction == "backward":
            await asyncio.wait_for(
                _pending_gradients_back.put(body),
                timeout=30.0,
            )
            log_event(f"P2P: received gradient {transfer_id[:12]}... from downstream ({body.get('source_miner', '?')})")

        return {"status": "received", "transfer_id": transfer_id, "queue_size": _pending_activations.qsize()}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Activation queue full — miner busy")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to receive activation: {e}")


@app.get("/p2p/serve-weights")
async def serve_weights(model_id: str = "", layer_start: int = 0, layer_end: int = 0):
    """
    Serve our loaded weight tensors to a requesting peer miner.
    
    This is the P2P weight sharing endpoint — when a new miner joins
    and needs layers we have, it pulls directly from us instead of
    going to the seed slicer. The model lives across the network.
    """
    import hashlib as _hashlib

    if _model_ref is None:
        raise HTTPException(status_code=503, detail="No model loaded yet")

    assignment = miner_state.get("shard_assignment")
    if not assignment:
        raise HTTPException(status_code=503, detail="Not in sharded mode")

    # Verify we actually have the requested layers
    our_start = assignment.get("layer_start", 0)
    our_end = assignment.get("layer_end", 0)
    if layer_start < our_start or layer_end > our_end:
        raise HTTPException(
            status_code=404,
            detail=f"We hold layers {our_start}-{our_end}, requested {layer_start}-{layer_end}"
        )

    # Serialize our model state_dict for transfer
    state = {}
    for name, param in _model_ref.named_parameters():
        state[name] = {
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "hash": _hashlib.sha256(param.detach().cpu().numpy().tobytes()).hexdigest()[:16],
        }

    return {
        "model_id": model_id or miner_state.get("model_id", ""),
        "layer_start": our_start,
        "layer_end": our_end,
        "num_params": sum(p.numel() for p in _model_ref.parameters()),
        "weight_hash": miner_state.get("_shard_weight_hash", ""),
        "params": state,
        "status": "available",
    }


@app.get("/p2p/status")
async def p2p_status():
    """Pipeline stage status — used by peers and server for health checks."""
    assignment = miner_state.get("shard_assignment")
    return {
        "miner_id": miner_state.get("miner_id"),
        "is_sharded": miner_state.get("is_sharded", False),
        "pipeline_group_id": miner_state.get("pipeline_group_id"),
        "stage_index": miner_state.get("stage_index"),
        "status": miner_state.get("status"),
        "pending_activations": _pending_activations.qsize(),
        "pending_gradients": _pending_gradients_back.qsize(),
        "layer_start": assignment.get("layer_start") if assignment else None,
        "layer_end": assignment.get("layer_end") if assignment else None,
        "upstream_peer": miner_state.get("upstream_peer"),
        "downstream_peer": miner_state.get("downstream_peer"),
        "model_loaded": _model_ref is not None,
    }


async def _send_activation_to_peer(peer_miner_id: str, payload: dict) -> bool:
    """Send compressed activation tensor to a pipeline peer via HTTP POST."""
    peer_url = _peer_addresses.get(peer_miner_id)
    if not peer_url:
        log_event(f"P2P: no address for peer {peer_miner_id}", "warning")
        return False

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{peer_url}/p2p/receive-activation",
                json=payload,
            )
            if resp.status_code == 200:
                return True
            else:
                log_event(f"P2P: peer {peer_miner_id} returned {resp.status_code}", "warning")
                return False
    except Exception as e:
        log_event(f"P2P: failed to send to {peer_miner_id}: {e}", "warning")
        return False


async def _wait_for_upstream_activation(timeout: float = 120.0) -> Optional[dict]:
    """Wait for activation from upstream miner. Returns None on timeout."""
    try:
        return await asyncio.wait_for(_pending_activations.get(), timeout=timeout)
    except asyncio.TimeoutError:
        log_event("P2P: timeout waiting for upstream activation", "warning")
        return None


async def _wait_for_downstream_gradient(timeout: float = 120.0) -> Optional[dict]:
    """Wait for gradient from downstream miner. Returns None on timeout."""
    try:
        return await asyncio.wait_for(_pending_gradients_back.get(), timeout=timeout)
    except asyncio.TimeoutError:
        log_event("P2P: timeout waiting for downstream gradient", "warning")
        return None


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


def _real_training_step(task: dict, cycle: int, model, data, device_str: str):
    """Run actual forward/backward pass. Returns grad_data + model/data refs."""
    import torch
    from real_trainer import get_tokenizer, DataShardLoader, RealTrainer, compress_gradients

    device = torch.device(device_str)

    # Scale parameters for device memory constraints
    # MPS (MacBook) and CPU: small batch + short seq to avoid OOM
    if device_str in ("mps", "cpu"):
        max_seq = 512
        bs = 1
    else:
        max_seq = task.get("max_seq_length", 4096)
        bs = task.get("batch_size", 8)

    # Init model on first call — shard-aware
    if model is None:
        assignment = miner_state.get("shard_assignment")
        model = _create_model_or_shard(task, assignment, device_str)

    # Init data on first call
    if data is None:
        tok = get_tokenizer()
        loader = DataShardLoader(tok, max_seq_length=max_seq)
        try:
            data = loader.load_from_huggingface(num_samples=max(50, bs * 20))
        except Exception:
            data = loader._generate_synthetic_data(max(50, bs * 20))

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
    compressed = compress_gradients(result["gradients"], top_k_ratio=0.0001)

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


async def _mining_loop(num_cycles: int):
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
    if not torch_ok:
        log_event("PyTorch not available — cannot mine without GPU training", "error")
        miner_state["status"] = "error"
        miner_state["error"] = "PyTorch required for mining"
        return
    miner_state["real_training"] = True
    mode = f"REAL on {dev_name}"

    log_event(f"Starting mining: {mode} | Model: {miner_state['model_id']} | {num_cycles} cycles")
    await broadcast({"type": "state", "data": _safe_state()})

    urls = _platform_urls()
    try:
        # Phase 1: Lighthouse
        log_event("Phase 1: Registering with Lighthouse...")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{urls['lighthouse']}/lighthouse/register",
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

        # Phase 1.5: Shard capability registration + assignment
        log_event("Phase 1.5: Registering shard capability...")
        await _register_capability(miner_id, token)
        assignment = await _fetch_shard_assignment(miner_id, token)
        if assignment:
            miner_state["shard_assignment"] = assignment
            miner_state["pipeline_group_id"] = assignment.get("pipeline_group_id")
            miner_state["stage_index"] = assignment.get("stage_index")
            miner_state["is_sharded"] = True
            miner_state["upstream_peer"] = assignment.get("upstream_miner_id")
            miner_state["downstream_peer"] = assignment.get("downstream_miner_id")
            log_event(
                f"Pipeline mode: stage {assignment['stage_index']}/{assignment['num_stages']} | "
                f"layers {assignment['layer_start']}-{assignment['layer_end']} | "
                f"upstream={assignment.get('upstream_miner_id', 'None')} | "
                f"downstream={assignment.get('downstream_miner_id', 'None')}"
            )
            # Fetch peer addresses for P2P activation routing
            peers_info = await _fetch_pipeline_peers(miner_id, token)
            for p in peers_info.get("peers", []):
                _peer_addresses[p["miner_id"]] = f"http://{p['address']}:{p['api_port']}"
            log_event(f"Pipeline peers discovered: {len(_peer_addresses)} addresses")

            # Phase 1.6: Network-native weight loading
            log_event("Phase 1.6: Requesting weight transfer plan...")
            transfer_plan = await _request_weight_transfer_plan(miner_id, token, assignment)
            if transfer_plan:
                peer_sources = [s for s in transfer_plan.get("sources", []) if s.get("type") == "peer"]
                if peer_sources:
                    log_event(f"Found {len(peer_sources)} peers with our layers — attempting P2P download")
                    for source in peer_sources:
                        addr = source.get("address", "")
                        if addr:
                            weights = await _download_weights_from_peer(
                                addr, assignment.get("model_id", ""),
                                assignment["layer_start"], assignment["layer_end"]
                            )
                            if weights:
                                log_event(f"P2P weight download succeeded from {addr}")
                                miner_state["_weight_source"] = f"peer:{addr}"
                                break
                    else:
                        log_event("No P2P peer reachable — will initialize from scratch")
                        miner_state["_weight_source"] = "genesis"
                else:
                    log_event("No peer sources in plan — will initialize from scratch")
                    miner_state["_weight_source"] = "genesis"
            else:
                miner_state["_weight_source"] = "genesis"
        else:
            log_event("No shard assignment — full model training mode")
            miner_state["is_sharded"] = False

        await broadcast({"type": "state", "data": _safe_state()})

        # Phase 2: Genesis
        log_event("Phase 2: Checking genesis...")
        try:
            async with httpx.AsyncClient(timeout=10, headers={"Authorization": f"Bearer {token}"}) as client:
                resp = await client.get(f"{urls['mining']}/mining/genesis/status")
                if resp.status_code == 200:
                    gs = resp.json()
                    if gs.get("genesis", {}).get("initialized"):
                        log_event("Genesis already initialized")
                    else:
                        log_event("Initializing genesis seed...")
                        await client.post(f"{urls['mining']}/mining/genesis/initialize", json={"model_id": miner_state["model_id"], "miner_ids": [miner_id], "ipfs_base_url": "ipfs://"})
                        log_event("Genesis initialized")
                else:
                    log_event(f"Genesis status: HTTP {resp.status_code}", "warning")
        except Exception as e:
            log_event(f"Genesis check: {e}", "warning")

        # Phase 3: Mining cycles (reconnect WS per cycle — server closes after each gradient)
        miner_state["status"] = "training"
        await broadcast({"type": "state", "data": _safe_state()})
        log_event("Phase 3: Starting mining cycles...")

        ws_url = f"{urls['ws_mining']}?token={token}"
        model = None
        data = None

        for cycle in range(1, num_cycles + 1):
            if miner_state["status"] != "training":
                break

            log_event(f"━━━ Cycle {cycle}/{num_cycles} ━━━")
            await broadcast({"type": "cycle", "current": cycle, "total": num_cycles})

            try:
                async with websockets.connect(ws_url, close_timeout=10, ping_interval=None) as ws:
                    # Register
                    await ws.send(json.dumps({"action": "register", "miner_id": miner_id, "miner_class": "validator_miner", "account_email": email}))
                    welcome = json.loads(await ws.recv())

                    if welcome.get("event") != "welcome":
                        log_event(f"Registration failed: {welcome}", "error")
                        break

                    if cycle == 1:
                        log_event(f"Connected: {miner_id} | Global step: {welcome.get('param_server', {}).get('global_step', 0)}")

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

                    # Train (real GPU training only — no simulation)
                    train_start = time.time()
                    log_event(f"Training on {dev_name}...")
                    try:
                        grad_data, model, data = await asyncio.to_thread(
                            _real_training_step, task, cycle, model, data, dev,
                        )
                    except Exception as e:
                        log_event(f"Training failed: {e}", "error")
                        miner_state["error"] = str(e)
                        await broadcast({"type": "error", "message": f"Training failed: {e}"})
                        continue

                    global _model_ref
                    _model_ref = model  # Persist for download

                    # Report loaded weights to registry on first cycle
                    if cycle == 1 and miner_state.get("is_sharded") and miner_state.get("shard_assignment"):
                        await _report_shard_loaded(miner_id, token, miner_state["shard_assignment"])
                        await _report_shard_ready(miner_id, token)

                    elapsed = time.time() - train_start
                    loss_b = grad_data["loss_before"]
                    loss_a = grad_data["loss_after"]
                    miner_state["current_loss"] = loss_a
                    miner_state["loss_history"].append({"cycle": cycle, "loss_before": loss_b, "loss_after": loss_a, "time": elapsed})

                    # Store detailed metrics for metrics panel
                    miner_state["detailed_metrics"].append({
                        "cycle": cycle,
                        "epoch": task.get("epoch", 0),
                        "batch_index": task.get("batch_index", 0),
                        "loss_before": loss_b,
                        "loss_after": loss_a,
                        "loss_delta": round(loss_b - loss_a, 6),
                        "grad_norm": grad_data.get("grad_norm", 0),
                        "learning_rate": task.get("learning_rate", 3e-4),
                        "training_time": elapsed,
                        "samples_processed": grad_data.get("samples_processed", 0),
                        "original_size": grad_data.get("original_size", 0),
                        "compressed_size": grad_data.get("compressed_size", 0),
                        "compression_ratio": grad_data.get("compression_ratio", 0),
                        "gradient_hash": grad_data.get("gradient_hash", "")[:16],
                        "device": grad_data.get("device", ""),
                        "throughput": round(grad_data.get("samples_processed", 0) / max(elapsed, 0.01), 1),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

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

                        # Auto-save checkpoint locally
                        ckpt_path = _save_checkpoint(model, miner_state["model_id"], cycle)
                        if ckpt_path:
                            log_event(f"💾 Checkpoint saved: ~/.rg_miner/checkpoints/{Path(ckpt_path).name}")

                        try:
                            agg_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            agg = json.loads(agg_raw)
                            if agg.get("event") == "aggregation_complete":
                                log_event(f"⚡ Aggregation — global step {agg['global_step']} ({agg['layers_merged']} layers)")
                        except (asyncio.TimeoutError, Exception):
                            pass

                    elif submit_msg.get("event") == "gradient_rejected":
                        log_event(f"✗ Gradient REJECTED: {submit_msg.get('reason')}", "error")
                    else:
                        log_event(f"Unexpected: {submit_msg}", "error")

                    await broadcast({"type": "state", "data": _safe_state()})

            except Exception as e:
                log_event(f"Cycle {cycle} connection error: {e}", "warning")

            # Pause between cycles — mining service needs time to finish aggregation
            if cycle < num_cycles:
                log_event("Waiting for aggregation to complete...")
                await asyncio.sleep(10)

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
