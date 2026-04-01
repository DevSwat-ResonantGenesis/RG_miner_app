"""
RESONANT MODEL ARCHITECTURE
=============================

Production Transformer architecture for decentralized training.
Implements modern LLM best practices:
  - Grouped Query Attention (GQA) — fewer KV heads, saves memory
  - Rotary Position Embeddings (RoPE) — no learned position embeddings
  - SwiGLU activation — better than ReLU/GELU for LLMs
  - RMSNorm — faster than LayerNorm
  - Pre-norm architecture — more stable training

Compatible with the MODEL_REGISTRY tiers (1B → 405B).
Designed to run on consumer GPUs (bf16, gradient checkpointing).

STATUS: PRODUCTION
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# MODEL CONFIG
# ══════════════════════════════════════════════════════════════

@dataclass
class ResonantModelConfig:
    """Model configuration matching MODEL_REGISTRY entries."""
    model_id: str = "resonant-seed-1b"
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 4
    intermediate_size: int = 5504
    vocab_size: int = 128_256
    max_seq_length: int = 4096
    dtype: str = "bfloat16"
    dropout: float = 0.0
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.dtype == "bfloat16" else torch.float32

    @classmethod
    def from_registry(cls, registry_entry: dict) -> "ResonantModelConfig":
        """Create config from a MODEL_REGISTRY entry."""
        return cls(
            model_id=registry_entry.get("model_id", "resonant-seed-1b"),
            hidden_size=registry_entry["hidden_size"],
            num_layers=registry_entry["num_layers"],
            num_heads=registry_entry["num_heads"],
            num_kv_heads=registry_entry.get("num_kv_heads", registry_entry["num_heads"] // 4),
            intermediate_size=registry_entry.get("intermediate_size", registry_entry["hidden_size"] * 4),
            vocab_size=registry_entry.get("vocab_size", 128_256),
            max_seq_length=registry_entry.get("max_seq_length", 4096),
        )


# ══════════════════════════════════════════════════════════════
# RMSNorm — faster than LayerNorm, used in LLaMA/Mistral
# ══════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ══════════════════════════════════════════════════════════════
# Rotary Position Embeddings (RoPE)
# ══════════════════════════════════════════════════════════════

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 500000.0) -> torch.Tensor:
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.shape[1], :].unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ══════════════════════════════════════════════════════════════
# Grouped Query Attention (GQA)
# ══════════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    """
    GQA: Multiple query heads share fewer KV heads.
    For 16 heads with 4 KV heads: each KV head serves 4 query heads.
    Saves ~40% memory vs MHA with minimal quality loss.
    """

    def __init__(self, config: ResonantModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q, k = apply_rope(q.transpose(1, 2), k.transpose(1, 2), freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Expand KV heads to match query heads (GQA)
        if self.num_groups > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1).reshape(bsz, seqlen, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1).reshape(bsz, seqlen, self.num_heads, self.head_dim)

        # Transpose to (bsz, num_heads, seqlen, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (uses FlashAttention when available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=mask is None,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(attn_output)


# ══════════════════════════════════════════════════════════════
# SwiGLU Feed-Forward Network
# ══════════════════════════════════════════════════════════════

class SwiGLUFFN(nn.Module):
    """
    SwiGLU: gate * silu(x) * up(x) — used in LLaMA, Mistral, etc.
    Better than GELU for LLMs (Shazeer 2020).
    """

    def __init__(self, config: ResonantModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ══════════════════════════════════════════════════════════════
# Transformer Block (Pre-Norm)
# ══════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    def __init__(self, config: ResonantModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLUFFN(config)
        self.attention_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm + attention + residual
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        # Pre-norm + FFN + residual
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


# ══════════════════════════════════════════════════════════════
# RESONANT MODEL — Full Transformer
# ══════════════════════════════════════════════════════════════

class ResonantModel(nn.Module):
    """
    The Resonant LLM — a modern Transformer for decentralized training.

    Architecture: GQA + RoPE + SwiGLU + RMSNorm + Pre-norm
    Comparable to LLaMA 3 / Mistral architecture.
    """

    def __init__(self, config: ResonantModelConfig):
        super().__init__()
        self.config = config

        # Token embedding (no position embedding — RoPE handles it)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size)

        # LM head (optionally tied to embedding)
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(config.head_dim, config.max_seq_length, config.rope_theta),
            persistent=False,
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Log param count
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"ResonantModel initialized: {config.model_id} | {total_params:,} parameters | {config.num_layers} layers")

    def _init_weights(self, module):
        """Initialize weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            labels: (batch_size, seq_len) target IDs for loss computation

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        bsz, seqlen = input_ids.shape

        # Token embeddings
        h = self.embed_tokens(input_ids)

        # Transformer blocks
        for layer in self.layers:
            h = layer(h, self.freqs_cis)

        # Final norm
        h = self.norm(h)

        # LM head
        if self.lm_head is not None:
            logits = self.lm_head(h)
        else:
            logits = F.linear(h, self.embed_tokens.weight)

        result = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def get_num_params(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: ResonantModelConfig) -> "ResonantModel":
        """Create model from config, cast to correct dtype."""
        model = cls(config)
        if config.torch_dtype == torch.bfloat16:
            model = model.to(dtype=torch.bfloat16)
        return model


# ══════════════════════════════════════════════════════════════
# FACTORY — Create model from registry
# ══════════════════════════════════════════════════════════════

def create_model(model_id: str, registry: dict = None) -> Tuple[ResonantModel, ResonantModelConfig]:
    """
    Create a ResonantModel from the model registry.

    Args:
        model_id: Key from MODEL_REGISTRY (e.g. "resonant-seed-1b")
        registry: Optional registry dict override

    Returns:
        (model, config) tuple
    """
    if registry is None:
        # Inline registry for standalone miner app (no genesis_seed module)
        registry = {
            "resonant-seed-1b": {
                "d_model": 2048, "num_layers": 24, "num_heads": 16,
                "num_kv_heads": 4, "d_ff": 5504, "vocab_size": 100277,
                "max_seq_length": 4096, "dropout": 0.0, "rope_theta": 10000.0,
                "description": "1B param seed model", "parameters": "~1B",
                "min_miners": 1,
            },
            "resonant-v1-7b": {
                "d_model": 4096, "num_layers": 32, "num_heads": 32,
                "num_kv_heads": 8, "d_ff": 11008, "vocab_size": 100277,
                "max_seq_length": 8192, "dropout": 0.0, "rope_theta": 10000.0,
                "description": "7B param v1 model", "parameters": "~7B",
                "min_miners": 4,
            },
            "resonant-v1-13b": {
                "d_model": 5120, "num_layers": 40, "num_heads": 40,
                "num_kv_heads": 8, "d_ff": 13824, "vocab_size": 100277,
                "max_seq_length": 8192, "dropout": 0.0, "rope_theta": 10000.0,
                "description": "13B param v1 model", "parameters": "~13B",
                "min_miners": 8,
            },
        }

    if model_id not in registry:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(registry.keys())}")

    entry = registry[model_id]
    entry["model_id"] = model_id
    config = ResonantModelConfig.from_registry(entry)
    model = ResonantModel.from_config(config)

    return model, config
