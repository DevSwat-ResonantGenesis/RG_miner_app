"""
1F1B Microbatch Execution Engine — Pipeline-Parallel Training on the Miner
============================================================================

Executes the 1F1B (One-Forward-One-Backward) interleaved schedule locally
on the miner's GPU. This is the "brain" that keeps the GPU saturated while
waiting for activations from pipeline peers.

The engine receives a schedule (list of actions) from the PipelineCoordinator
on the server and executes them in order:

  Warmup:      F0  F1  F2  F3           ← fill pipeline with forwards
  Steady:      B0  F4  B1  F5  B2  F6   ← interleave forward + backward
  Cooldown:    B3  B4  B5  B6  B7       ← drain remaining backwards

For each forward pass:
  Stage 0:     input_ids → embed → layers → send activation downstream
  Middle:      receive activation → layers → send activation downstream
  Last stage:  receive activation → layers → norm → logits → compute loss

For each backward pass:
  Last stage:  loss.backward() from saved computation graph
  Middle:      receive grad from downstream → backward through saved graph
  Stage 0:     receive grad from downstream → backward → accumulate

After all microbatches: compress + submit accumulated gradients.

Peak activation memory: num_stages tensors (vs num_microbatches for GPipe).
"""

import asyncio
import base64
import hashlib
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger("rg-miner.microbatch-engine")


# ══════════════════════════════════════════════════════════════
# SCHEDULE ACTIONS (mirrors server-side pipeline.py)
# ══════════════════════════════════════════════════════════════

class Action:
    FORWARD = "forward"
    BACKWARD = "backward"
    IDLE = "idle"
    SUBMIT_GRADIENTS = "submit_gradients"
    SYNC = "sync"


@dataclass
class ScheduleStep:
    """One step in the pipeline schedule for this stage."""
    step_index: int
    stage_index: int
    action: str
    microbatch_index: int = -1
    clock_tick: int = 0
    depends_on: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduleStep":
        return cls(
            step_index=d.get("step_index", 0),
            stage_index=d.get("stage_index", 0),
            action=d.get("action", "idle"),
            microbatch_index=d.get("microbatch_index", -1),
            clock_tick=d.get("clock_tick", 0),
            depends_on=d.get("depends_on"),
        )


# ══════════════════════════════════════════════════════════════
# MICROBATCH STATE
# ══════════════════════════════════════════════════════════════

@dataclass
class MicrobatchSlot:
    """Tracks one microbatch through this stage's forward and backward."""
    index: int
    # Forward pass saved state (for backward)
    input_activation: Any = None    # The tensor we received (detached, requires_grad)
    output_activation: Any = None   # What we computed (with grad_fn attached)
    labels: Any = None              # Only for last stage
    loss: Optional[float] = None
    # Timing
    forward_start: float = 0.0
    forward_end: float = 0.0
    backward_start: float = 0.0
    backward_end: float = 0.0
    forward_done: bool = False
    backward_done: bool = False


@dataclass
class EngineStats:
    """Execution statistics for one training step."""
    execution_id: str = ""
    stage_index: int = 0
    num_microbatches: int = 0
    total_forwards: int = 0
    total_backwards: int = 0
    total_idles: int = 0
    forward_time: float = 0.0
    backward_time: float = 0.0
    idle_time: float = 0.0
    p2p_send_time: float = 0.0
    p2p_recv_time: float = 0.0
    avg_loss: float = 0.0
    peak_memory_mb: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

    @property
    def gpu_utilization(self) -> float:
        """Fraction of time spent on actual compute (vs waiting)."""
        compute = self.forward_time + self.backward_time
        return compute / self.total_time if self.total_time > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "stage_index": self.stage_index,
            "num_microbatches": self.num_microbatches,
            "total_forwards": self.total_forwards,
            "total_backwards": self.total_backwards,
            "total_idles": self.total_idles,
            "forward_time_sec": round(self.forward_time, 3),
            "backward_time_sec": round(self.backward_time, 3),
            "idle_time_sec": round(self.idle_time, 3),
            "p2p_send_time_sec": round(self.p2p_send_time, 3),
            "p2p_recv_time_sec": round(self.p2p_recv_time, 3),
            "total_time_sec": round(self.total_time, 3),
            "gpu_utilization": round(self.gpu_utilization, 4),
            "avg_loss": round(self.avg_loss, 6),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
        }


# ══════════════════════════════════════════════════════════════
# ACTIVATION SERIALIZATION — for P2P tensor transfer
# ══════════════════════════════════════════════════════════════

def serialize_activation(tensor) -> dict:
    """Serialize a PyTorch tensor for network transfer."""
    import torch

    # Detach from computation graph and move to CPU for serialization
    t = tensor.detach().cpu()
    buf = io.BytesIO()
    torch.save(t, buf)
    raw = buf.getvalue()

    return {
        "data": base64.b64encode(raw).decode("ascii"),
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "size_bytes": len(raw),
        "hash": hashlib.sha256(raw).hexdigest()[:16],
    }


def deserialize_activation(payload: dict, device: str = "cpu"):
    """Deserialize a tensor from network transfer."""
    import torch

    raw = base64.b64decode(payload["data"])
    buf = io.BytesIO(raw)
    t = torch.load(buf, map_location=device, weights_only=True)
    return t


# ══════════════════════════════════════════════════════════════
# 1F1B MICROBATCH ENGINE
# ══════════════════════════════════════════════════════════════

class MicrobatchEngine:
    """
    Executes a 1F1B pipeline schedule on this miner's GPU.

    The engine manages:
    1. Splitting the batch into microbatches
    2. Executing forward/backward passes per the schedule
    3. Saving activations for backward pass (gradient checkpointing)
    4. Sending/receiving activations and gradients via P2P callbacks
    5. Accumulating gradients across microbatches
    6. Compressing and returning final gradients

    Usage:
        engine = MicrobatchEngine(model, device, stage_index, num_stages)
        engine.set_p2p_callbacks(send_fn, recv_activation_fn, recv_gradient_fn)
        result = await engine.execute(schedule_steps, input_ids, labels, lr)
    """

    def __init__(
        self,
        model,
        device: str,
        stage_index: int,
        num_stages: int,
        miner_id: str = "",
    ):
        self.model = model
        self.device = device
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.miner_id = miner_id

        self.is_first_stage = (stage_index == 0)
        self.is_last_stage = (stage_index == num_stages - 1)

        # P2P callbacks (set by caller)
        self._send_activation: Optional[Callable] = None
        self._send_gradient: Optional[Callable] = None
        self._recv_activation: Optional[Callable] = None
        self._recv_gradient: Optional[Callable] = None

        # State
        self._slots: Dict[int, MicrobatchSlot] = {}
        self._accumulated_grads: Dict[str, Any] = {}
        self._stats = EngineStats()

    def set_p2p_callbacks(
        self,
        send_activation: Callable,
        send_gradient: Callable,
        recv_activation: Callable,
        recv_gradient: Callable,
    ):
        """
        Set async callbacks for P2P tensor transfer.

        send_activation(payload: dict) -> bool
            Send activation to downstream peer. Payload contains serialized tensor.
        send_gradient(payload: dict) -> bool
            Send gradient to upstream peer.
        recv_activation(timeout: float) -> Optional[dict]
            Wait for activation from upstream peer. Returns serialized tensor.
        recv_gradient(timeout: float) -> Optional[dict]
            Wait for gradient from downstream peer.
        """
        self._send_activation = send_activation
        self._send_gradient = send_gradient
        self._recv_activation = recv_activation
        self._recv_gradient = recv_gradient

    async def execute(
        self,
        schedule_steps: List[dict],
        input_data=None,
        labels_data=None,
        learning_rate: float = 3e-4,
        num_microbatches: int = 4,
    ) -> Dict[str, Any]:
        """
        Execute a full 1F1B schedule for one training step.

        Args:
            schedule_steps: List of step dicts from the PipelineCoordinator
            input_data: Full batch of input_ids (stage 0 only)
            labels_data: Full batch of labels (last stage only, or all stages for loss)
            learning_rate: For the optimizer
            num_microbatches: How many microbatches to split the batch into

        Returns:
            Dict with gradient data, losses, and execution stats
        """
        import torch

        execution_id = f"exec-{uuid4().hex[:8]}"
        self._stats = EngineStats(
            execution_id=execution_id,
            stage_index=self.stage_index,
            num_microbatches=num_microbatches,
            start_time=time.time(),
        )
        self._slots.clear()
        self._accumulated_grads.clear()

        # Parse schedule
        steps = [ScheduleStep.from_dict(s) for s in schedule_steps]

        # Split input data into microbatches (stage 0 only)
        microbatch_inputs = []
        microbatch_labels = []
        if input_data is not None:
            mb_size = max(1, input_data.size(0) // num_microbatches)
            for i in range(num_microbatches):
                start = i * mb_size
                end = min(start + mb_size, input_data.size(0))
                if start >= input_data.size(0):
                    start = 0
                    end = mb_size
                microbatch_inputs.append(input_data[start:end])
        if labels_data is not None:
            mb_size = max(1, labels_data.size(0) // num_microbatches)
            for i in range(num_microbatches):
                start = i * mb_size
                end = min(start + mb_size, labels_data.size(0))
                if start >= labels_data.size(0):
                    start = 0
                    end = mb_size
                microbatch_labels.append(labels_data[start:end])

        # Initialize optimizer for gradient accumulation
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=0.01
        )
        optimizer.zero_grad()

        # Execute schedule step by step
        losses = []
        for step in steps:
            if step.action == Action.FORWARD:
                await self._execute_forward(
                    step, microbatch_inputs, microbatch_labels
                )
            elif step.action == Action.BACKWARD:
                loss = await self._execute_backward(step)
                if loss is not None:
                    losses.append(loss)
            elif step.action == Action.IDLE:
                idle_start = time.time()
                await asyncio.sleep(0.001)  # Yield to event loop
                self._stats.idle_time += time.time() - idle_start
                self._stats.total_idles += 1
            elif step.action == Action.SUBMIT_GRADIENTS:
                pass  # Handled below

        # Average gradients across microbatches
        if num_microbatches > 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(num_microbatches)

        # Optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )
        optimizer.step()
        optimizer.zero_grad()

        # Compute loss after update (for the last microbatch, quick eval)
        loss_after = None
        if self.is_last_stage and microbatch_labels:
            self.model.eval()
            with torch.no_grad():
                last_mb_idx = min(len(microbatch_inputs) - 1, num_microbatches - 1)
                if self.is_first_stage and microbatch_inputs:
                    # Single-stage: full model
                    result = self.model(
                        hidden_states=None,
                        input_ids=microbatch_inputs[last_mb_idx],
                        labels=microbatch_labels[last_mb_idx],
                    )
                    loss_after = result.get("loss", result.get("total_loss"))
                    if loss_after is not None:
                        loss_after = loss_after.item()
            self.model.train()

        # Collect gradients for compression + submission
        all_grads = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                all_grads.append(param.grad.detach().cpu().flatten())
        grad_vector = torch.cat(all_grads) if all_grads else torch.tensor([])

        # Track peak memory
        if torch.cuda.is_available():
            self._stats.peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6

        self._stats.end_time = time.time()
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        self._stats.avg_loss = avg_loss

        logger.info(
            f"1F1B complete: stage {self.stage_index}/{self.num_stages} | "
            f"{self._stats.total_forwards}F/{self._stats.total_backwards}B | "
            f"GPU util={self._stats.gpu_utilization:.1%} | "
            f"loss={avg_loss:.4f} | {self._stats.total_time:.1f}s"
        )

        return {
            "grad_vector": grad_vector,
            "grad_norm": float(grad_norm) if isinstance(grad_norm, (int, float)) else grad_norm.item(),
            "losses": losses,
            "loss_before": losses[0] if losses else 0.0,
            "loss_after": loss_after if loss_after is not None else (losses[-1] if losses else 0.0),
            "avg_loss": avg_loss,
            "num_microbatches": num_microbatches,
            "stats": self._stats.to_dict(),
            "execution_id": execution_id,
        }

    # ── Forward pass ──────────────────────────────────────────

    async def _execute_forward(
        self,
        step: ScheduleStep,
        microbatch_inputs: List,
        microbatch_labels: List,
    ):
        """Execute one forward microbatch on our shard."""
        import torch

        mb_idx = step.microbatch_index
        slot = MicrobatchSlot(index=mb_idx)
        slot.forward_start = time.time()

        self.model.train()

        if self.is_first_stage:
            # Stage 0: embed input_ids
            if mb_idx < len(microbatch_inputs):
                input_ids = microbatch_inputs[mb_idx].to(self.device)
            else:
                input_ids = microbatch_inputs[0].to(self.device)

            # Need grad on input for backward pass
            result = self.model(
                hidden_states=None,
                input_ids=input_ids,
                labels=microbatch_labels[mb_idx].to(self.device) if (
                    self.is_last_stage and mb_idx < len(microbatch_labels)
                ) else None,
            )
            slot.input_activation = input_ids
        else:
            # Middle/last stage: receive activation from upstream
            recv_start = time.time()
            upstream_payload = await self._recv_activation(timeout=120.0)
            self._stats.p2p_recv_time += time.time() - recv_start

            if upstream_payload is None:
                logger.error(f"Forward mb={mb_idx}: timeout waiting for upstream activation")
                slot.forward_done = True
                self._slots[mb_idx] = slot
                return

            hidden = deserialize_activation(upstream_payload, self.device)
            hidden.requires_grad_(True)
            slot.input_activation = hidden

            result = self.model(
                hidden_states=hidden,
                labels=microbatch_labels[mb_idx].to(self.device) if (
                    self.is_last_stage and mb_idx < len(microbatch_labels)
                ) else None,
            )

        # Save output for backward pass
        slot.output_activation = result.get("hidden_states")

        # Last stage: save loss
        if self.is_last_stage and "loss" in result:
            slot.loss = result["loss"].item()
            slot.labels = microbatch_labels[mb_idx] if mb_idx < len(microbatch_labels) else None
            # Keep computation graph alive for backward
            slot.output_activation = result.get("total_loss", result["loss"])

        # Send activation downstream (if not last stage)
        if not self.is_last_stage and slot.output_activation is not None:
            send_start = time.time()
            payload = serialize_activation(slot.output_activation)
            payload["microbatch_index"] = mb_idx
            payload["source_stage"] = self.stage_index
            payload["direction"] = "forward"
            payload["source_miner"] = self.miner_id
            payload["transfer_id"] = f"act-{uuid4().hex[:10]}"
            await self._send_activation(payload)
            self._stats.p2p_send_time += time.time() - send_start

        slot.forward_done = True
        slot.forward_end = time.time()
        self._stats.forward_time += slot.forward_end - slot.forward_start
        self._stats.total_forwards += 1
        self._slots[mb_idx] = slot

    # ── Backward pass ─────────────────────────────────────────

    async def _execute_backward(self, step: ScheduleStep) -> Optional[float]:
        """Execute one backward microbatch. Returns loss if last stage."""
        import torch

        mb_idx = step.microbatch_index
        slot = self._slots.get(mb_idx)
        if not slot:
            logger.error(f"Backward mb={mb_idx}: no saved forward state")
            return None

        slot.backward_start = time.time()
        loss_val = slot.loss

        if self.is_last_stage:
            # Last stage: backward from loss
            loss_tensor = slot.output_activation
            if loss_tensor is not None and loss_tensor.requires_grad:
                loss_tensor.backward(retain_graph=False)
        else:
            # Middle/first stage: receive gradient from downstream
            recv_start = time.time()
            downstream_payload = await self._recv_gradient(timeout=120.0)
            self._stats.p2p_recv_time += time.time() - recv_start

            if downstream_payload is None:
                logger.error(f"Backward mb={mb_idx}: timeout waiting for downstream gradient")
                slot.backward_done = True
                return loss_val

            grad_tensor = deserialize_activation(downstream_payload, self.device)

            # Backward through our computation graph
            output = slot.output_activation
            if output is not None and output.requires_grad:
                output.backward(gradient=grad_tensor, retain_graph=False)

        # Send gradient to upstream (if not first stage)
        if not self.is_first_stage and slot.input_activation is not None:
            if slot.input_activation.grad is not None:
                send_start = time.time()
                payload = serialize_activation(slot.input_activation.grad)
                payload["microbatch_index"] = mb_idx
                payload["source_stage"] = self.stage_index
                payload["direction"] = "backward"
                payload["source_miner"] = self.miner_id
                payload["transfer_id"] = f"grad-{uuid4().hex[:10]}"
                await self._send_gradient(payload)
                self._stats.p2p_send_time += time.time() - send_start

        # Free saved activations to reduce memory
        slot.input_activation = None
        slot.output_activation = None
        slot.labels = None

        slot.backward_done = True
        slot.backward_end = time.time()
        self._stats.backward_time += slot.backward_end - slot.backward_start
        self._stats.total_backwards += 1

        return loss_val

    # ── Utilities ─────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return self._stats.to_dict()


# ══════════════════════════════════════════════════════════════
# SCHEDULE GENERATOR (local mirror of server-side pipeline.py)
# ══════════════════════════════════════════════════════════════

def generate_local_1f1b_schedule(
    stage_index: int,
    num_stages: int,
    num_microbatches: int,
) -> List[dict]:
    """
    Generate the 1F1B schedule steps for THIS stage only.

    Used when the miner doesn't receive a schedule from the server
    (e.g., during local testing or when running standalone).

    Returns list of step dicts ready for engine.execute().
    """
    steps = []
    step_counter = 0
    tick = stage_index

    # Warmup: forward passes to fill the pipeline
    warmup_count = min(num_stages - stage_index - 1, num_microbatches)
    forward_mb = 0
    backward_mb = 0

    for _ in range(warmup_count):
        if forward_mb >= num_microbatches:
            break
        steps.append({
            "step_index": step_counter,
            "stage_index": stage_index,
            "action": Action.FORWARD,
            "microbatch_index": forward_mb,
            "clock_tick": tick,
        })
        step_counter += 1
        forward_mb += 1
        tick += 1

    # Steady state: 1 forward + 1 backward alternating
    while forward_mb < num_microbatches:
        steps.append({
            "step_index": step_counter,
            "stage_index": stage_index,
            "action": Action.FORWARD,
            "microbatch_index": forward_mb,
            "clock_tick": tick,
        })
        step_counter += 1
        forward_mb += 1
        tick += 1

        if backward_mb < num_microbatches:
            steps.append({
                "step_index": step_counter,
                "stage_index": stage_index,
                "action": Action.BACKWARD,
                "microbatch_index": backward_mb,
                "clock_tick": tick,
            })
            step_counter += 1
            backward_mb += 1
            tick += 1

    # Cooldown: remaining backward passes
    while backward_mb < num_microbatches:
        steps.append({
            "step_index": step_counter,
            "stage_index": stage_index,
            "action": Action.BACKWARD,
            "microbatch_index": backward_mb,
            "clock_tick": tick,
        })
        step_counter += 1
        backward_mb += 1
        tick += 1

    # Submit gradients
    steps.append({
        "step_index": step_counter,
        "stage_index": stage_index,
        "action": Action.SUBMIT_GRADIENTS,
        "clock_tick": tick,
    })

    return steps
