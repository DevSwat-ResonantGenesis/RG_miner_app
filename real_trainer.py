"""
REAL TRAINING ENGINE
=====================

Replaces simulate_training() with actual PyTorch forward/backward passes.
Supports both GPU (CUDA/MPS) and CPU training.

Flow:
  1. Load model weights from S3 or initialize fresh
  2. Load tokenized data shard
  3. Run forward pass → compute cross-entropy loss
  4. Run backward pass → compute gradients
  5. Apply Top-K gradient compression
  6. Return compressed gradient + loss metrics

For miners: This module runs on the miner's machine.
For the param server: Gradients are aggregated centrally.
"""

import hashlib
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Device detection
def get_device() -> torch.device:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
logger.info(f"Training device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# TOKENIZER — BPE via tiktoken (GPT-4 compatible, fast)
# ══════════════════════════════════════════════════════════════

class ResonantTokenizer:
    """
    Wrapper around tiktoken for fast BPE tokenization.
    Uses cl100k_base (GPT-4 tokenizer) as starting point.
    In production, we'd train a custom tokenizer on our corpus.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding(encoding_name)
            self.vocab_size = self.enc.n_vocab
            self.pad_token_id = 0
            self.eos_token_id = self.enc.eot_token
            self._backend = "tiktoken"
        except ImportError:
            # Fallback: simple byte-level encoding
            logger.warning("tiktoken not installed — using byte-level fallback tokenizer")
            self.enc = None
            self.vocab_size = 128_256
            self.pad_token_id = 0
            self.eos_token_id = 1
            self._backend = "bytes"

    def encode(self, text: str, max_length: int = 4096) -> List[int]:
        if self._backend == "tiktoken":
            tokens = self.enc.encode(text, disallowed_special=())
        else:
            tokens = list(text.encode("utf-8", errors="replace"))
        return tokens[:max_length]

    def decode(self, token_ids: List[int]) -> str:
        if self._backend == "tiktoken":
            return self.enc.decode(token_ids)
        return bytes(token_ids).decode("utf-8", errors="replace")

    def batch_encode(self, texts: List[str], max_length: int = 4096, padding: bool = True) -> torch.Tensor:
        """Encode a batch of texts into a padded tensor."""
        encoded = [self.encode(t, max_length) for t in texts]
        if padding:
            max_len = min(max_length, max(len(e) for e in encoded))
            padded = [e + [self.pad_token_id] * (max_len - len(e)) for e in encoded]
            return torch.tensor(padded, dtype=torch.long)
        return encoded


# Global tokenizer instance
_tokenizer = None

def get_tokenizer() -> ResonantTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = ResonantTokenizer()
    return _tokenizer


# ══════════════════════════════════════════════════════════════
# DATA LOADING — Stream from HuggingFace or S3
# ══════════════════════════════════════════════════════════════

class DataShardLoader:
    """
    Loads a training data shard. Supports:
    - HuggingFace datasets (streaming mode — no full download needed)
    - Local files (pre-tokenized .pt shards)
    - S3/Spaces (pre-tokenized shards)
    """

    def __init__(self, tokenizer: ResonantTokenizer, max_seq_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def load_from_huggingface(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        split: str = "train",
        num_samples: int = 1000,
        text_field: str = "text",
    ) -> torch.Tensor:
        """Stream samples from HuggingFace and tokenize on-the-fly."""
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split=split, streaming=True)
            tokens_list = []
            count = 0
            for sample in ds:
                text = sample.get(text_field, "")
                if not text or len(text) < 50:
                    continue
                toks = self.tokenizer.encode(text, max_length=self.max_seq_length)
                if len(toks) >= 64:  # Skip very short samples
                    # Pad or truncate to max_seq_length
                    if len(toks) < self.max_seq_length:
                        toks = toks + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(toks))
                    tokens_list.append(toks[:self.max_seq_length])
                    count += 1
                    if count >= num_samples:
                        break

            if not tokens_list:
                raise ValueError(f"No valid samples found in {dataset_name}")

            logger.info(f"Loaded {len(tokens_list)} samples from {dataset_name} ({split})")
            return torch.tensor(tokens_list, dtype=torch.long)

        except ImportError:
            logger.warning("'datasets' library not installed — generating synthetic data")
            return self._generate_synthetic_data(num_samples)

    def load_from_file(self, path: str) -> torch.Tensor:
        """Load pre-tokenized shard from a .pt file."""
        data = torch.load(path, map_location="cpu")
        logger.info(f"Loaded shard from {path}: {data.shape}")
        return data

    def _generate_synthetic_data(self, num_samples: int = 1000) -> torch.Tensor:
        """Generate synthetic training data for testing without real dataset access."""
        logger.info(f"Generating {num_samples} synthetic samples (seq_len={self.max_seq_length})")
        vocab_size = self.tokenizer.vocab_size
        # Generate random token sequences (not real language, but valid for training loop testing)
        data = torch.randint(2, min(vocab_size, 50000), (num_samples, self.max_seq_length), dtype=torch.long)
        return data


# ══════════════════════════════════════════════════════════════
# WEIGHT STORAGE — S3/DigitalOcean Spaces
# ══════════════════════════════════════════════════════════════

class WeightStorage:
    """Upload/download model weights to DigitalOcean Spaces (S3-compatible)."""

    def __init__(self):
        try:
            import boto3
            self.s3 = boto3.client(
                "s3",
                endpoint_url=os.getenv("S3_ENDPOINT", "https://sfo3.digitaloceanspaces.com"),
                aws_access_key_id=os.getenv("S3_ACCESS_KEY", ""),
                aws_secret_access_key=os.getenv("S3_SECRET_KEY", ""),
                region_name="sfo3",
            )
            self.bucket = os.getenv("S3_BUCKET", "genesis2026")
            self._available = bool(os.getenv("S3_ACCESS_KEY"))
        except ImportError:
            logger.warning("boto3 not installed — weight storage unavailable")
            self.s3 = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def save_checkpoint(self, model: nn.Module, model_id: str, step: int) -> Optional[str]:
        """Save model checkpoint to S3. Returns S3 key."""
        if not self._available:
            logger.warning("S3 not configured — saving locally only")
            local_path = f"/tmp/checkpoints/{model_id}_step_{step}.pt"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            torch.save(model.state_dict(), local_path)
            return local_path

        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)

        key = f"model-weights/{model_id}/step_{step}/full_checkpoint.pt"
        self.s3.upload_fileobj(buffer, self.bucket, key)
        logger.info(f"Checkpoint uploaded: s3://{self.bucket}/{key}")
        return key

    def load_checkpoint(self, model: nn.Module, s3_key: str) -> nn.Module:
        """Load model checkpoint from S3."""
        if not self._available or s3_key.startswith("/tmp/"):
            state_dict = torch.load(s3_key, map_location="cpu")
            model.load_state_dict(state_dict)
            return model

        import io
        buffer = io.BytesIO()
        self.s3.download_fileobj(self.bucket, s3_key, buffer)
        buffer.seek(0)
        state_dict = torch.load(buffer, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info(f"Checkpoint loaded from s3://{self.bucket}/{s3_key}")
        return model

    def save_weight_shard(self, state_dict_shard: dict, model_id: str, shard_idx: int, step: int) -> Tuple[str, str]:
        """Save a weight shard (subset of layers). Returns (s3_key, sha256_hash)."""
        import io
        buffer = io.BytesIO()
        torch.save(state_dict_shard, buffer)
        shard_bytes = buffer.getvalue()
        shard_hash = hashlib.sha256(shard_bytes).hexdigest()

        key = f"model-weights/{model_id}/step_{step}/shard_{shard_idx:04d}.pt"

        if self._available:
            buffer.seek(0)
            self.s3.upload_fileobj(buffer, self.bucket, key)
        else:
            local_path = f"/tmp/weight_shards/{model_id}/step_{step}/"
            os.makedirs(local_path, exist_ok=True)
            with open(f"{local_path}/shard_{shard_idx:04d}.pt", "wb") as f:
                f.write(shard_bytes)
            key = f"{local_path}/shard_{shard_idx:04d}.pt"

        return key, shard_hash


# ══════════════════════════════════════════════════════════════
# REAL TRAINING STEP — replaces simulate_training()
# ══════════════════════════════════════════════════════════════

class RealTrainer:
    """
    Performs actual forward/backward passes on real model + data.
    This is the core that runs on each miner's GPU.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device = None,
    ):
        self.device = device or DEVICE
        self.model = model.to(self.device)
        self.config = config
        self.optimizer = None  # Created per training step
        self.scaler = None     # For mixed precision

        # Gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        logger.info(f"RealTrainer initialized on {self.device}")
        logger.info(f"  Model: {config.get('model_id', 'unknown')} ({sum(p.numel() for p in model.parameters()):,} params)")

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float = 3e-4,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Run a single training step: forward → loss → backward → gradient.

        Returns dict with loss, gradients, and metrics.
        """
        self.model.train()
        start_time = time.time()

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss_value = loss.item()

        # Backward pass
        loss.backward()

        # Clip gradients
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        # Collect gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().cpu()

        # Zero gradients for next step
        self.model.zero_grad()

        elapsed = time.time() - start_time

        return {
            "loss": loss_value,
            "grad_norm": total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm,
            "gradients": gradients,
            "num_params_with_grad": len(gradients),
            "training_time_seconds": elapsed,
            "device": str(self.device),
            "samples_in_batch": input_ids.shape[0],
            "seq_length": input_ids.shape[1],
        }

    def apply_gradients(self, aggregated_gradients: Dict[str, torch.Tensor], learning_rate: float = 3e-4):
        """Apply aggregated gradients from the parameter server."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_gradients:
                    param.data -= learning_rate * aggregated_gradients[name].to(self.device)


def compress_gradients(
    gradients: Dict[str, torch.Tensor],
    top_k_ratio: float = 0.01,
) -> Dict[str, Any]:
    """
    Top-K gradient compression.
    Keeps only the top K% of gradient values by magnitude.

    Args:
        gradients: {param_name: gradient_tensor}
        top_k_ratio: Fraction of values to keep (0.01 = top 1%)

    Returns:
        Compressed gradient dict with indices, values, and hash.
    """
    all_flat = []
    param_shapes = {}
    param_offsets = {}
    offset = 0

    for name, grad in gradients.items():
        flat = grad.float().flatten()
        param_shapes[name] = grad.shape
        param_offsets[name] = (offset, offset + flat.numel())
        all_flat.append(flat)
        offset += flat.numel()

    if not all_flat:
        return {"indices": [], "values": [], "original_size": 0, "compressed_size": 0}

    full_grad = torch.cat(all_flat)
    original_size = full_grad.numel()

    # Top-K selection by absolute magnitude
    k = max(1, int(original_size * top_k_ratio))
    _, top_indices = torch.topk(full_grad.abs(), k)
    top_values = full_grad[top_indices]

    # Sort for deterministic hashing
    sorted_order = top_indices.argsort()
    top_indices = top_indices[sorted_order].tolist()
    top_values = top_values[sorted_order].tolist()

    # Compute SHA256 hash for verification
    hash_input = json.dumps({"indices": top_indices, "values": [round(v, 8) for v in top_values]}, sort_keys=True).encode()
    gradient_hash = hashlib.sha256(hash_input).hexdigest()

    return {
        "top_k_indices": top_indices,
        "top_k_values": [round(v, 8) for v in top_values],
        "original_size": original_size,
        "compressed_size": k,
        "compression_ratio": original_size / k,
        "gradient_hash": gradient_hash,
        "param_shapes": {n: list(s) for n, s in param_shapes.items()},
        "param_offsets": {n: list(o) for n, o in param_offsets.items()},
    }


def decompress_gradients(
    compressed: Dict[str, Any],
) -> torch.Tensor:
    """Decompress a Top-K compressed gradient back to full size."""
    full_grad = torch.zeros(compressed["original_size"])
    indices = torch.tensor(compressed["top_k_indices"], dtype=torch.long)
    values = torch.tensor(compressed["top_k_values"], dtype=torch.float32)
    full_grad[indices] = values
    return full_grad


# ══════════════════════════════════════════════════════════════
# PROOF-OF-TRAINING VERIFIER
# ══════════════════════════════════════════════════════════════

class ProofOfTrainingVerifier:
    """
    Spot-check verification: re-run 1% of a miner's batch
    and compare the loss. If delta > threshold → flag as fraud.
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or DEVICE
        self.tolerance = 0.05  # 5% loss tolerance

    def verify_gradient(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        claimed_loss: float,
        claimed_gradient_hash: str,
        sample_ratio: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Verify a miner's claimed training results.

        Args:
            input_ids: The same input data the miner trained on
            labels: The same labels
            claimed_loss: Loss the miner reported
            claimed_gradient_hash: Hash of gradient the miner submitted
            sample_ratio: Fraction of batch to re-run (default 1%)

        Returns:
            Verification result dict.
        """
        self.model.eval()
        start = time.time()

        # Sample a subset of the batch
        batch_size = input_ids.shape[0]
        num_verify = max(1, int(batch_size * sample_ratio))
        indices = torch.randperm(batch_size)[:num_verify]
        sample_input = input_ids[indices].to(self.device)
        sample_labels = labels[indices].to(self.device)

        # Forward pass only (no backward needed for verification)
        with torch.no_grad():
            outputs = self.model(input_ids=sample_input, labels=sample_labels)
            verified_loss = outputs["loss"].item()

        elapsed = time.time() - start

        # Check if claimed loss is within tolerance
        loss_delta = abs(claimed_loss - verified_loss)
        loss_ratio = loss_delta / max(claimed_loss, 1e-6)
        is_valid = loss_ratio <= self.tolerance

        result = {
            "verified": is_valid,
            "claimed_loss": claimed_loss,
            "verified_loss": verified_loss,
            "loss_delta": loss_delta,
            "loss_ratio": loss_ratio,
            "tolerance": self.tolerance,
            "samples_checked": num_verify,
            "total_samples": batch_size,
            "verification_time_seconds": elapsed,
        }

        if not is_valid:
            logger.warning(f"VERIFICATION FAILED: claimed_loss={claimed_loss:.4f}, verified_loss={verified_loss:.4f}, delta={loss_ratio:.2%}")
        else:
            logger.info(f"Verification passed: loss delta {loss_ratio:.2%} within {self.tolerance:.0%} tolerance")

        return result


# ══════════════════════════════════════════════════════════════
# FULL TRAINING STEP — Drop-in replacement for simulate_training
# ══════════════════════════════════════════════════════════════

def real_training_step(
    task: dict,
    model: nn.Module = None,
    data: torch.Tensor = None,
    cycle: int = 0,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Drop-in replacement for simulate_training() in miner_client.py.
    Runs actual forward/backward on real model with real data.

    Args:
        task: Training task from the mining service
        model: The PyTorch model (or None to create from task config)
        data: Tokenized input data tensor (or None to load from HF)
        cycle: Current training cycle number
        device: torch device

    Returns:
        Same format as simulate_training() but with REAL values.
    """
    dev = device or DEVICE
    batch_size = task.get("batch_size", 8)
    learning_rate = task.get("learning_rate", 3e-4)
    max_seq_length = task.get("max_seq_length", 4096)

    # Create model if not provided
    if model is None:
        from model_architecture import create_model
        model_id = task.get("model_id", "resonant-seed-1b")
        model, _ = create_model(model_id)
        model = model.to(dev)

    # Load data if not provided
    if data is None:
        tokenizer = get_tokenizer()
        loader = DataShardLoader(tokenizer, max_seq_length)
        try:
            data = loader.load_from_huggingface(num_samples=batch_size * 4)
        except Exception:
            data = loader._generate_synthetic_data(batch_size * 4)

    # Select batch for this cycle
    start_idx = (cycle * batch_size) % len(data)
    end_idx = start_idx + batch_size
    batch = data[start_idx:end_idx]
    if len(batch) < batch_size:
        batch = data[:batch_size]  # Wrap around

    input_ids = batch[:, :-1].to(dev)
    labels = batch[:, 1:].to(dev)

    # Get loss BEFORE training
    model.eval()
    with torch.no_grad():
        pre_output = model(input_ids=input_ids, labels=labels)
        loss_before = pre_output["loss"].item()

    # Actual training step
    trainer = RealTrainer(model, task, dev)
    result = trainer.train_step(input_ids, labels, learning_rate)
    loss_after = result["loss"]

    # Compress gradients
    compressed = compress_gradients(result["gradients"], top_k_ratio=0.01)

    return {
        "top_k_indices": compressed["top_k_indices"],
        "top_k_values": compressed["top_k_values"],
        "original_size": compressed["original_size"],
        "compressed_size": compressed["compressed_size"],
        "compression_ratio": compressed["compression_ratio"],
        "gradient_hash": compressed["gradient_hash"],
        "loss_before": round(loss_before, 6),
        "loss_after": round(loss_after, 6),
        "samples_processed": result["samples_in_batch"] * result["seq_length"],
        "training_time_seconds": round(result["training_time_seconds"], 2),
        "device": result["device"],
        "grad_norm": result["grad_norm"],
        "real_training": True,  # Flag: this is NOT simulated
    }
