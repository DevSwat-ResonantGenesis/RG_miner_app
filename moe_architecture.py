"""
Mixture of Experts (MoE) Architecture — Sparse FFN for Frontier-Scale Models
==============================================================================

Extends the existing ResonantModel architecture with MoE layers.
Instead of a single dense FFN per transformer block, each MoE block has
N expert FFN networks and a learned router that selects top-k experts
per token.

Key properties:
  - Total parameters: N × expert_size (e.g. 64 experts × 6.3B each ≈ 405B)
  - Active parameters per token: top_k × expert_size (e.g. 2 × 6.3B ≈ 12.6B)
  - Compute per token: ~8x less than equivalent dense model
  - Memory for weights: same as dense (all experts stored)
  - Memory for activations: much smaller (only active experts)

Uses the same GQA + RoPE + RMSNorm as the dense model. Only the FFN
is replaced with the MoE routing + sparse expert FFN.

References:
  - Switch Transformers (Fedus et al. 2021)
  - GShard (Lepikhin et al. 2020)
  - Mixtral (Jiang et al. 2024)
  - DeepSeek-MoE (Dai et al. 2024)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_architecture import (
    GroupedQueryAttention,
    RMSNorm,
    ResonantModelConfig,
    SwiGLUFFN,
    precompute_rope_freqs,
)

logger = logging.getLogger("rg-mining.moe")


# ══════════════════════════════════════════════════════════════
# MOE CONFIG
# ══════════════════════════════════════════════════════════════

@dataclass
class MoEConfig(ResonantModelConfig):
    """Extended config for Mixture of Experts models."""
    num_experts: int = 64            # Total number of expert FFNs
    num_experts_per_token: int = 2   # Top-k experts activated per token
    expert_intermediate_size: int = 0  # Per-expert FFN size (0 = auto from intermediate_size)
    num_shared_experts: int = 0      # Shared experts always active (DeepSeek-MoE style)
    load_balance_alpha: float = 0.01 # Auxiliary loss weight for load balancing
    router_jitter: float = 0.0      # Jitter noise for router during training
    expert_capacity_factor: float = 1.25  # Capacity factor for expert buffers

    # MoE layer placement — not every layer needs to be MoE
    # e.g. for 126 layers, every 2nd layer is MoE = 63 MoE layers
    moe_layer_frequency: int = 2     # Every Nth layer is MoE (1 = all layers)

    @property
    def effective_expert_intermediate(self) -> int:
        if self.expert_intermediate_size > 0:
            return self.expert_intermediate_size
        # Default: same intermediate_size divided by a factor to keep total params similar
        # For Mixtral-style: each expert has full intermediate_size
        return self.intermediate_size

    @property
    def total_expert_params(self) -> int:
        """Total parameters across all experts (one layer)."""
        expert_params = self.hidden_size * self.effective_expert_intermediate * 3  # gate+up+down
        return self.num_experts * expert_params

    @property
    def active_expert_params(self) -> int:
        """Active parameters per token (one layer)."""
        expert_params = self.hidden_size * self.effective_expert_intermediate * 3
        return self.num_experts_per_token * expert_params

    @property
    def sparsity_ratio(self) -> float:
        """Fraction of parameters active per token."""
        return self.num_experts_per_token / self.num_experts

    @classmethod
    def from_registry_moe(cls, registry_entry: dict) -> "MoEConfig":
        """Create MoE config from a MODEL_REGISTRY entry."""
        config = cls(
            model_id=registry_entry.get("model_id", "resonant-frontier-405b"),
            hidden_size=registry_entry["hidden_size"],
            num_layers=registry_entry["num_layers"],
            num_heads=registry_entry["num_heads"],
            num_kv_heads=registry_entry.get("num_kv_heads", registry_entry["num_heads"] // 4),
            intermediate_size=registry_entry.get("intermediate_size", registry_entry["hidden_size"] * 4),
            vocab_size=registry_entry.get("vocab_size", 128_256),
            max_seq_length=registry_entry.get("max_seq_length", 32768),
            # MoE-specific
            num_experts=registry_entry.get("num_experts", 64),
            num_experts_per_token=registry_entry.get("num_experts_per_token", 2),
            expert_intermediate_size=registry_entry.get("expert_intermediate_size", 0),
            num_shared_experts=registry_entry.get("num_shared_experts", 0),
            moe_layer_frequency=registry_entry.get("moe_layer_frequency", 2),
        )
        return config

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "model_id": self.model_id,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_seq_length": self.max_seq_length,
        }
        moe = {
            "num_experts": self.num_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "effective_expert_intermediate": self.effective_expert_intermediate,
            "num_shared_experts": self.num_shared_experts,
            "moe_layer_frequency": self.moe_layer_frequency,
            "load_balance_alpha": self.load_balance_alpha,
            "total_expert_params_per_layer": self.total_expert_params,
            "active_expert_params_per_layer": self.active_expert_params,
            "sparsity_ratio": round(self.sparsity_ratio, 4),
        }
        return {**base, **moe}


# ══════════════════════════════════════════════════════════════
# EXPERT ROUTER — Learned top-k gating
# ══════════════════════════════════════════════════════════════

class ExpertRouter(nn.Module):
    """
    Top-k router that assigns each token to the best k experts.
    
    The router is a simple linear layer that produces logits over experts.
    Top-k selection determines which experts process each token.
    
    Includes:
    - Load balancing auxiliary loss (prevents expert collapse)
    - Optional jitter noise during training (improves exploration)
    - Capacity factor to handle uneven routing
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.jitter = config.router_jitter
        self.alpha = config.load_balance_alpha

        # Router: hidden_size → num_experts
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: [batch, seq, hidden]
            
        Returns:
            router_weights: [batch, seq, top_k] — softmax weights for selected experts
            selected_experts: [batch, seq, top_k] — indices of selected experts
            router_loss: scalar — auxiliary load balancing loss
        """
        batch_size, seq_len, hidden = hidden_states.shape

        # Optional jitter for training exploration
        if self.training and self.jitter > 0:
            noise = torch.randn_like(hidden_states) * self.jitter
            router_input = hidden_states + noise
        else:
            router_input = hidden_states

        # Compute router logits: [batch, seq, num_experts]
        router_logits = self.gate(router_input)

        # Top-k selection
        router_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        # Normalize weights via softmax (only over selected experts)
        router_weights = F.softmax(router_weights, dim=-1)

        # Load balancing loss
        router_loss = self._load_balance_loss(router_logits, selected_experts)

        return router_weights, selected_experts, router_loss

    def _load_balance_loss(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.
        
        Encourages even distribution of tokens across experts.
        Without this, the router tends to collapse to using only a few experts.
        
        L_balance = alpha * N * sum(f_i * P_i) for i in experts
        where:
          f_i = fraction of tokens routed to expert i
          P_i = average router probability for expert i
          N = num_experts
          
        This is the standard Switch Transformer balance loss.
        """
        if self.alpha == 0:
            return torch.tensor(0.0, device=router_logits.device)

        # Flatten batch and seq dims
        num_tokens = router_logits.shape[0] * router_logits.shape[1]
        router_probs = F.softmax(router_logits, dim=-1)  # [batch, seq, num_experts]

        # f_i: fraction of tokens routed to each expert
        # Count how many times each expert is selected
        expert_mask = F.one_hot(selected_experts, self.num_experts)  # [..., top_k, num_experts]
        expert_mask = expert_mask.sum(dim=-2)  # [..., num_experts] — count per token
        expert_mask = expert_mask.float()
        
        tokens_per_expert = expert_mask.sum(dim=[0, 1])  # [num_experts]
        f_i = tokens_per_expert / max(num_tokens, 1)

        # P_i: average router probability for each expert
        P_i = router_probs.mean(dim=[0, 1])  # [num_experts]

        # Balance loss
        loss = self.alpha * self.num_experts * (f_i * P_i).sum()

        return loss


# ══════════════════════════════════════════════════════════════
# EXPERT FFN — Individual expert network
# ══════════════════════════════════════════════════════════════

class ExpertFFN(nn.Module):
    """
    Single expert FFN — same SwiGLU architecture as dense model.
    Each expert is a full SwiGLU FFN with its own weights.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ══════════════════════════════════════════════════════════════
# MOE LAYER — Router + N Experts
# ══════════════════════════════════════════════════════════════

class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    
    Replaces the single dense FFN with:
    1. A router that selects top-k experts per token
    2. N expert FFNs (each with their own weights)
    3. Optional shared experts that always contribute
    
    The output is a weighted sum of the selected experts' outputs.
    
    For 405B with 64 experts and top-2:
    - Each expert: ~6.3B params (hidden=16384, intermediate=53248)
    - Active per token: 2 experts = ~12.6B params
    - Total FFN params: 64 × 6.3B = ~403B (the majority of the model)
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.hidden_size = config.hidden_size

        # Router
        self.router = ExpertRouter(config)

        # Expert FFNs
        expert_intermediate = config.effective_expert_intermediate
        self.experts = nn.ModuleList([
            ExpertFFN(config.hidden_size, expert_intermediate)
            for _ in range(config.num_experts)
        ])

        # Optional shared experts (always active, not routed)
        self.shared_experts = None
        if config.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                ExpertFFN(config.hidden_size, expert_intermediate)
                for _ in range(config.num_shared_experts)
            ])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: [batch, seq, hidden]
            
        Returns:
            output: [batch, seq, hidden] — weighted sum of expert outputs
            router_loss: scalar — auxiliary balance loss
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Route tokens to experts
        router_weights, selected_experts, router_loss = self.router(hidden_states)

        # Initialize output
        output = torch.zeros_like(hidden_states)

        # Process each expert
        # For efficiency, batch all tokens going to the same expert
        flat_hidden = hidden_states.view(-1, hidden_dim)  # [B*S, H]
        flat_weights = router_weights.view(-1, self.top_k)  # [B*S, top_k]
        flat_experts = selected_experts.view(-1, self.top_k)  # [B*S, top_k]
        flat_output = torch.zeros_like(flat_hidden)  # [B*S, H]

        for k in range(self.top_k):
            expert_indices = flat_experts[:, k]  # [B*S]
            expert_weights = flat_weights[:, k].unsqueeze(-1)  # [B*S, 1]

            for expert_idx in range(self.num_experts):
                # Find tokens assigned to this expert at position k
                mask = (expert_indices == expert_idx)
                if not mask.any():
                    continue

                # Gather tokens for this expert
                expert_input = flat_hidden[mask]  # [num_tokens, H]
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_input)

                # Add weighted output
                flat_output[mask] += expert_output * expert_weights[mask]

        output = flat_output.view(batch_size, seq_len, hidden_dim)

        # Add shared expert contributions (if any)
        if self.shared_experts is not None:
            shared_weight = 1.0 / (self.top_k + len(self.shared_experts))
            for shared_expert in self.shared_experts:
                output = output + shared_expert(hidden_states) * shared_weight

        return output, router_loss


# ══════════════════════════════════════════════════════════════
# MOE TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════

class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE FFN instead of dense FFN.
    
    Structure:
      Input → RMSNorm → GQA Attention → Residual
            → RMSNorm → MoE(expert FFNs) → Residual → Output
    """

    def __init__(self, config: MoEConfig, layer_idx: int, use_moe: bool = True):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_moe = use_moe

        # Attention (same as dense model)
        self.attention = GroupedQueryAttention(config)
        self.attention_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)

        # FFN: MoE or dense based on layer position
        if use_moe:
            self.moe = MoELayer(config)
            self.feed_forward = None
        else:
            self.moe = None
            self.feed_forward = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            hidden_states: [batch, seq, hidden]
            router_loss: scalar (0 if dense layer)
        """
        # Pre-norm + attention + residual
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)

        # Pre-norm + FFN (MoE or dense) + residual
        normed = self.ffn_norm(x)
        if self.use_moe and self.moe is not None:
            ffn_out, router_loss = self.moe(normed)
        else:
            ffn_out = self.feed_forward(normed)
            router_loss = torch.tensor(0.0, device=x.device)

        x = x + ffn_out
        return x, router_loss


# ══════════════════════════════════════════════════════════════
# RESONANT MOE MODEL — Full MoE Transformer
# ══════════════════════════════════════════════════════════════

class ResonantMoEModel(nn.Module):
    """
    Resonant LLM with Mixture of Experts — frontier-scale architecture.
    
    Same backbone as ResonantModel (GQA + RoPE + RMSNorm + Pre-norm)
    but with MoE FFN layers for massive parameter scaling with
    controlled compute cost.
    
    MoE layer placement:
    - Every Nth layer is MoE (configurable via moe_layer_frequency)
    - Non-MoE layers use standard dense SwiGLU FFN
    - This mixed approach (Mixtral/DeepSeek style) balances quality and efficiency
    
    For resonant-frontier-405b:
    - 126 layers, every 2nd is MoE = 63 MoE + 63 dense
    - 64 experts per MoE layer, top-2 routing
    - ~405B total params, ~50B active per token
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks — alternating dense and MoE
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            use_moe = (config.moe_layer_frequency > 0 and
                       i % config.moe_layer_frequency == 0)
            self.layers.append(MoETransformerBlock(config, layer_idx=i, use_moe=use_moe))

        # Final norm
        self.norm = RMSNorm(config.hidden_size)

        # LM head
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(config.head_dim, config.max_seq_length, config.rope_theta),
            persistent=False,
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Stats
        total_params = sum(p.numel() for p in self.parameters())
        num_moe_layers = sum(1 for l in self.layers if l.use_moe)
        num_dense_layers = config.num_layers - num_moe_layers
        logger.info(
            f"ResonantMoEModel initialized: {config.model_id} | "
            f"{total_params:,} total params | "
            f"{num_moe_layers} MoE + {num_dense_layers} dense layers | "
            f"{config.num_experts} experts × top-{config.num_experts_per_token}"
        )

    def _init_weights(self, module):
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
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Returns dict with:
          - logits: [batch, seq, vocab]
          - loss: cross-entropy loss (if labels provided)
          - router_loss: sum of all MoE layers' balance losses
          - total_loss: loss + router_loss
        """
        bsz, seqlen = input_ids.shape
        h = self.embed_tokens(input_ids)

        # Accumulate router losses
        total_router_loss = torch.tensor(0.0, device=input_ids.device)

        for layer in self.layers:
            h, router_loss = layer(h, self.freqs_cis)
            total_router_loss = total_router_loss + router_loss

        h = self.norm(h)

        if self.lm_head is not None:
            logits = self.lm_head(h)
        else:
            logits = F.linear(h, self.embed_tokens.weight)

        result = {
            "logits": logits,
            "router_loss": total_router_loss,
        }

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss
            result["total_loss"] = loss + total_router_loss

        return result

    def get_num_params(self, count_all: bool = True) -> Dict[str, int]:
        """
        Parameter count breakdown.
        
        Returns dict with total, active-per-token, embedding, attention, expert counts.
        """
        total = sum(p.numel() for p in self.parameters())
        embedding = sum(p.numel() for p in self.embed_tokens.parameters())
        
        attention_params = 0
        expert_params = 0
        dense_ffn_params = 0
        router_params = 0
        
        for layer in self.layers:
            attention_params += sum(p.numel() for p in layer.attention.parameters())
            if layer.use_moe and layer.moe is not None:
                router_params += sum(p.numel() for p in layer.moe.router.parameters())
                expert_params += sum(p.numel() for p in layer.moe.experts.parameters())
                if layer.moe.shared_experts is not None:
                    expert_params += sum(p.numel() for p in layer.moe.shared_experts.parameters())
            elif layer.feed_forward is not None:
                dense_ffn_params += sum(p.numel() for p in layer.feed_forward.parameters())

        # Active params per token: all non-expert params + top-k experts per MoE layer
        active_expert_per_layer = 0
        if self.config.num_experts > 0:
            active_expert_per_layer = (
                expert_params / max(sum(1 for l in self.layers if l.use_moe), 1)
                * self.config.num_experts_per_token / self.config.num_experts
            )
        num_moe_layers = sum(1 for l in self.layers if l.use_moe)
        active_per_token = total - expert_params + int(active_expert_per_layer * num_moe_layers)

        return {
            "total": total,
            "active_per_token": active_per_token,
            "embedding": embedding,
            "attention": attention_params,
            "expert_ffn": expert_params,
            "dense_ffn": dense_ffn_params,
            "router": router_params,
            "sparsity_ratio": round(active_per_token / max(total, 1), 4),
        }

    @classmethod
    def from_config(cls, config: MoEConfig) -> "ResonantMoEModel":
        model = cls(config)
        if config.torch_dtype == torch.bfloat16:
            model = model.to(dtype=torch.bfloat16)
        return model


# ══════════════════════════════════════════════════════════════
# MODEL SHARD — A slice of a model for pipeline parallelism
# ══════════════════════════════════════════════════════════════

class ModelShard(nn.Module):
    """
    A contiguous slice of transformer layers for pipeline parallelism.
    
    Each miner in a pipeline group holds one ModelShard containing:
    - A subset of transformer layers (e.g. layers 32-63)
    - Optionally the embedding layer (stage 0 only)
    - Optionally the LM head + final norm (last stage only)
    
    Works with both dense (ResonantModel) and MoE (ResonantMoEModel) layers.
    
    This is what miners download and hold in memory — NOT the full model.
    For 405B with 8 stages, each shard is ~50B params ≈ ~100GB in fp16.
    """

    def __init__(
        self,
        config: ResonantModelConfig,
        layer_start: int,
        layer_end: int,
        has_embedding: bool = False,
        has_lm_head: bool = False,
        is_moe: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.has_embedding = has_embedding
        self.has_lm_head = has_lm_head
        self.is_moe = is_moe

        # Embedding (only for first stage)
        self.embed_tokens = None
        if has_embedding:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers for our range
        self.layers = nn.ModuleList()
        for i in range(layer_start, layer_end):
            if is_moe and isinstance(config, MoEConfig):
                use_moe = (config.moe_layer_frequency > 0 and
                           i % config.moe_layer_frequency == 0)
                self.layers.append(MoETransformerBlock(config, layer_idx=i, use_moe=use_moe))
            else:
                from model_architecture import TransformerBlock
                self.layers.append(TransformerBlock(config, layer_idx=i))

        # Final norm + LM head (only for last stage)
        self.norm = None
        self.lm_head = None
        if has_lm_head:
            self.norm = RMSNorm(config.hidden_size)
            if not config.tie_word_embeddings:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(config.head_dim, config.max_seq_length, config.rope_theta),
            persistent=False,
        )

        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"ModelShard: layers {layer_start}-{layer_end} | "
            f"{total_params:,} params | "
            f"embed={has_embedding} lm_head={has_lm_head} moe={is_moe}"
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through this shard.
        
        For stage 0 (has_embedding): pass input_ids, returns hidden_states
        For middle stages: pass hidden_states, returns hidden_states
        For last stage (has_lm_head): pass hidden_states, returns logits + loss
        """
        total_router_loss = torch.tensor(0.0, device=hidden_states.device if hidden_states is not None else input_ids.device)

        # Stage 0: embed tokens
        if self.has_embedding and input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)

        # Process through our layers
        for layer in self.layers:
            if isinstance(layer, MoETransformerBlock):
                hidden_states, rloss = layer(hidden_states, self.freqs_cis)
                total_router_loss = total_router_loss + rloss
            else:
                hidden_states = layer(hidden_states, self.freqs_cis)

        result = {
            "hidden_states": hidden_states,
            "router_loss": total_router_loss,
        }

        # Last stage: norm + LM head + loss
        if self.has_lm_head:
            h = self.norm(hidden_states) if self.norm else hidden_states
            if self.lm_head is not None:
                logits = self.lm_head(h)
            elif self.embed_tokens is not None:
                logits = F.linear(h, self.embed_tokens.weight)
            else:
                logits = h  # Shouldn't happen in practice
            result["logits"] = logits

            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                result["loss"] = loss
                result["total_loss"] = loss + total_router_loss

        return result


# ══════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ══════════════════════════════════════════════════════════════

def create_moe_model(
    model_id: str,
    registry: dict = None,
) -> Tuple[ResonantMoEModel, MoEConfig]:
    """Create a MoE model from the registry."""
    if registry is None:
        from genesis_seed import MODEL_REGISTRY
        registry = MODEL_REGISTRY

    if model_id not in registry:
        raise ValueError(f"Unknown model: {model_id}")

    entry = registry[model_id]
    entry["model_id"] = model_id

    if entry.get("model_type") != "transformer-gqa-moe":
        raise ValueError(f"Model {model_id} is not MoE type (got {entry.get('model_type')})")

    config = MoEConfig.from_registry_moe(entry)
    model = ResonantMoEModel.from_config(config)
    return model, config


def create_model_shard(
    model_id: str,
    layer_start: int,
    layer_end: int,
    has_embedding: bool = False,
    has_lm_head: bool = False,
    registry: dict = None,
    model_config: dict = None,
) -> Tuple[ModelShard, ResonantModelConfig]:
    """
    Create a model shard for pipeline parallelism.
    
    Works with both dense and MoE models.
    Each miner calls this to create ONLY their portion of the model.
    
    Standalone miner version: accepts model_config dict directly so
    we don't need genesis_seed.py on the miner side.
    """
    entry = None

    if model_config is not None:
        entry = dict(model_config)
        entry.setdefault("model_id", model_id)
    elif registry is not None:
        if model_id not in registry:
            raise ValueError(f"Unknown model: {model_id}")
        entry = dict(registry[model_id])
        entry["model_id"] = model_id
    else:
        # Try importing from model_architecture (miner has this)
        try:
            from model_architecture import create_model, ResonantModelConfig as RMC
            # For dense models, build config from the existing create_model factory
            model, cfg = create_model(model_id)
            del model  # Don't need the full model
            entry = {
                "model_id": model_id,
                "model_type": "transformer-gqa",
                "hidden_size": cfg.hidden_size,
                "num_layers": cfg.num_layers,
                "num_heads": cfg.num_heads,
                "num_kv_heads": cfg.num_kv_heads,
                "intermediate_size": cfg.intermediate_size,
                "vocab_size": cfg.vocab_size,
                "max_seq_length": cfg.max_seq_length,
            }
        except Exception:
            raise ValueError(
                f"Cannot create shard for '{model_id}': no registry, no config dict, "
                f"and model_architecture.create_model() failed"
            )

    is_moe = entry.get("model_type") == "transformer-gqa-moe"

    if is_moe:
        config = MoEConfig.from_registry_moe(entry)
    else:
        config = ResonantModelConfig.from_registry(entry)

    shard = ModelShard(
        config=config,
        layer_start=layer_start,
        layer_end=layer_end,
        has_embedding=has_embedding,
        has_lm_head=has_lm_head,
        is_moe=is_moe,
    )

    return shard, config
