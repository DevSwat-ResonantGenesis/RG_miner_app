"""
Tests for 1F1B Microbatch Execution Engine
=============================================

Tests the MicrobatchEngine with:
  - Single-stage pipeline (no P2P, full model)
  - Multi-stage pipeline with mock P2P callbacks
  - Schedule generation
  - Activation serialization round-trip
  - GPU utilization tracking
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available — skipping tests")
    sys.exit(0)

from microbatch_engine import (
    MicrobatchEngine,
    Action,
    ScheduleStep,
    MicrobatchSlot,
    EngineStats,
    serialize_activation,
    deserialize_activation,
    generate_local_1f1b_schedule,
)


# ══════════════════════════════════════════════════════════════
# Test helpers
# ══════════════════════════════════════════════════════════════

passed = 0
failed = 0
errors = []


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        import traceback
        print(f"  ✗ {name}: {e}")
        traceback.print_exc()
        failed += 1
        errors.append((name, str(e)))


class TinyModel(nn.Module):
    """Minimal model for testing — single linear layer with loss."""

    def __init__(self, vocab_size=100, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.linear = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, hidden_states=None, input_ids=None, labels=None):
        if input_ids is not None:
            hidden_states = self.embed(input_ids)
        h = self.linear(hidden_states)
        result = {"hidden_states": h}
        logits = self.head(h)
        result["logits"] = logits
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
            )
            result["loss"] = loss
            result["total_loss"] = loss
        return result


class TinyShardFirst(nn.Module):
    """First stage shard — embeds + linear."""

    def __init__(self, vocab_size=100, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, hidden_states=None, input_ids=None, labels=None):
        if input_ids is not None:
            hidden_states = self.embed(input_ids)
        h = self.linear(hidden_states)
        return {"hidden_states": h}


class TinyShardLast(nn.Module):
    """Last stage shard — linear + head + loss."""

    def __init__(self, vocab_size=100, hidden=32):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, hidden_states=None, input_ids=None, labels=None):
        h = self.linear(hidden_states)
        result = {"hidden_states": h}
        logits = self.head(h)
        result["logits"] = logits
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
            )
            result["loss"] = loss
            result["total_loss"] = loss
        return result


# ══════════════════════════════════════════════════════════════
# TEST 1: Schedule Generation
# ══════════════════════════════════════════════════════════════

def test_schedule_single_stage():
    steps = generate_local_1f1b_schedule(0, 1, 4)
    actions = [s["action"] for s in steps]
    # Single stage: all forwards, then all backwards, then submit
    forwards = [s for s in steps if s["action"] == Action.FORWARD]
    backwards = [s for s in steps if s["action"] == Action.BACKWARD]
    submits = [s for s in steps if s["action"] == Action.SUBMIT_GRADIENTS]
    assert len(forwards) == 4
    assert len(backwards) == 4
    assert len(submits) == 1


def test_schedule_multi_stage_first():
    """Stage 0 of 4 stages, 8 microbatches."""
    steps = generate_local_1f1b_schedule(0, 4, 8)
    forwards = [s for s in steps if s["action"] == Action.FORWARD]
    backwards = [s for s in steps if s["action"] == Action.BACKWARD]
    assert len(forwards) == 8
    assert len(backwards) == 8
    # Warmup: first 3 forwards (num_stages - stage - 1 = 3)
    warmup = steps[:3]
    assert all(s["action"] == Action.FORWARD for s in warmup)


def test_schedule_multi_stage_last():
    """Stage 3 of 4 stages, 8 microbatches — no warmup."""
    steps = generate_local_1f1b_schedule(3, 4, 8)
    forwards = [s for s in steps if s["action"] == Action.FORWARD]
    backwards = [s for s in steps if s["action"] == Action.BACKWARD]
    assert len(forwards) == 8
    assert len(backwards) == 8
    # Last stage: warmup=0, steady state starts immediately: F0 B0 F1 B1 ...
    assert steps[0]["action"] == Action.FORWARD
    assert steps[1]["action"] == Action.BACKWARD  # immediate backward after first forward


def test_schedule_microbatch_indices():
    steps = generate_local_1f1b_schedule(0, 2, 4)
    forward_mbs = [s["microbatch_index"] for s in steps if s["action"] == Action.FORWARD]
    backward_mbs = [s["microbatch_index"] for s in steps if s["action"] == Action.BACKWARD]
    assert forward_mbs == [0, 1, 2, 3]
    assert backward_mbs == [0, 1, 2, 3]


# ══════════════════════════════════════════════════════════════
# TEST 2: Activation Serialization
# ══════════════════════════════════════════════════════════════

def test_serialize_roundtrip():
    t = torch.randn(2, 4, 8)
    payload = serialize_activation(t)
    assert "data" in payload
    assert payload["shape"] == [2, 4, 8]
    assert payload["hash"] != ""
    t2 = deserialize_activation(payload, "cpu")
    assert torch.allclose(t, t2)


def test_serialize_different_dtypes():
    for dtype in [torch.float32, torch.float16]:
        t = torch.randn(3, 3).to(dtype)
        payload = serialize_activation(t)
        t2 = deserialize_activation(payload, "cpu")
        assert t2.shape == t.shape


# ══════════════════════════════════════════════════════════════
# TEST 3: Engine — Single Stage (full model, no P2P)
# ══════════════════════════════════════════════════════════════

def test_engine_single_stage():
    """Single-stage pipeline: engine runs full model with no P2P."""
    model = TinyModel(vocab_size=50, hidden=16)
    engine = MicrobatchEngine(
        model=model,
        device="cpu",
        stage_index=0,
        num_stages=1,
        miner_id="test-miner",
    )

    # No-op P2P callbacks (single stage doesn't need them)
    async def noop_send(payload): return True
    async def noop_recv(timeout=120.0): return None

    engine.set_p2p_callbacks(noop_send, noop_send, noop_recv, noop_recv)

    num_mb = 4
    schedule = generate_local_1f1b_schedule(0, 1, num_mb)

    input_ids = torch.randint(0, 50, (8, 16))
    labels = torch.randint(0, 50, (8, 16))

    result = asyncio.get_event_loop().run_until_complete(
        engine.execute(
            schedule_steps=schedule,
            input_data=input_ids,
            labels_data=labels,
            learning_rate=1e-3,
            num_microbatches=num_mb,
        )
    )

    assert "grad_vector" in result
    assert "losses" in result
    assert "stats" in result
    assert len(result["losses"]) == num_mb  # Each backward reports a loss
    assert result["stats"]["total_forwards"] == num_mb
    assert result["stats"]["total_backwards"] == num_mb
    assert result["grad_norm"] > 0
    assert result["stats"]["gpu_utilization"] > 0


def test_engine_single_stage_grad_update():
    """Verify that the model actually updates after engine execution."""
    model = TinyModel(vocab_size=50, hidden=16)
    before_params = {n: p.clone() for n, p in model.named_parameters()}

    engine = MicrobatchEngine(model=model, device="cpu", stage_index=0, num_stages=1)
    async def noop_send(payload): return True
    async def noop_recv(timeout=120.0): return None
    engine.set_p2p_callbacks(noop_send, noop_send, noop_recv, noop_recv)

    schedule = generate_local_1f1b_schedule(0, 1, 2)
    input_ids = torch.randint(0, 50, (4, 8))
    labels = torch.randint(0, 50, (4, 8))

    asyncio.get_event_loop().run_until_complete(
        engine.execute(schedule, input_ids, labels, learning_rate=0.1, num_microbatches=2)
    )

    # At least some parameters should have changed
    changed = 0
    for n, p in model.named_parameters():
        if not torch.allclose(before_params[n], p, atol=1e-7):
            changed += 1
    assert changed > 0, "Model parameters didn't update after training"


# ══════════════════════════════════════════════════════════════
# TEST 4: Engine — Two-Stage Pipeline with Mock P2P
# ══════════════════════════════════════════════════════════════

def test_engine_two_stage_pipeline():
    """
    Simulate a 2-stage pipeline: stage 0 (first) and stage 1 (last).
    Mock P2P by connecting their queues directly.
    """
    model_first = TinyShardFirst(vocab_size=50, hidden=16)
    model_last = TinyShardLast(vocab_size=50, hidden=16)

    # Shared queues for P2P
    activation_queue = asyncio.Queue()
    gradient_queue = asyncio.Queue()

    # Stage 0 engine
    engine0 = MicrobatchEngine(
        model=model_first, device="cpu",
        stage_index=0, num_stages=2, miner_id="miner-0",
    )

    # Stage 1 engine
    engine1 = MicrobatchEngine(
        model=model_last, device="cpu",
        stage_index=1, num_stages=2, miner_id="miner-1",
    )

    # Wire P2P: stage 0 sends activations → stage 1 receives
    #           stage 1 sends gradients → stage 0 receives
    async def send_to_stage1(payload):
        await activation_queue.put(payload)
        return True

    async def recv_from_stage0(timeout=120.0):
        return await asyncio.wait_for(activation_queue.get(), timeout=timeout)

    async def send_to_stage0(payload):
        await gradient_queue.put(payload)
        return True

    async def recv_from_stage1(timeout=120.0):
        return await asyncio.wait_for(gradient_queue.get(), timeout=timeout)

    async def noop_send(payload): return True
    async def noop_recv(timeout=120.0): return None

    engine0.set_p2p_callbacks(
        send_activation=send_to_stage1,    # Send forward activations to stage 1
        send_gradient=noop_send,           # Stage 0 has no upstream
        recv_activation=noop_recv,         # Stage 0 has no upstream
        recv_gradient=recv_from_stage1,    # Receive backward gradients from stage 1
    )

    engine1.set_p2p_callbacks(
        send_activation=noop_send,         # Stage 1 is last, no downstream
        send_gradient=send_to_stage0,      # Send backward gradients to stage 0
        recv_activation=recv_from_stage0,  # Receive forward activations from stage 0
        recv_gradient=noop_recv,           # Stage 1 has no downstream
    )

    num_mb = 4
    schedule0 = generate_local_1f1b_schedule(0, 2, num_mb)
    schedule1 = generate_local_1f1b_schedule(1, 2, num_mb)

    input_ids = torch.randint(0, 50, (8, 16))
    labels = torch.randint(0, 50, (8, 16))

    async def run_pipeline():
        # Run both stages concurrently
        result0, result1 = await asyncio.gather(
            engine0.execute(schedule0, input_data=input_ids, num_microbatches=num_mb),
            engine1.execute(schedule1, labels_data=labels, num_microbatches=num_mb),
        )
        return result0, result1

    r0, r1 = asyncio.get_event_loop().run_until_complete(run_pipeline())

    # Stage 0: should have done forward + backward for all microbatches
    assert r0["stats"]["total_forwards"] == num_mb
    assert r0["stats"]["total_backwards"] == num_mb
    assert r0["grad_norm"] > 0

    # Stage 1: should have done forward + backward with losses
    assert r1["stats"]["total_forwards"] == num_mb
    assert r1["stats"]["total_backwards"] == num_mb
    assert len(r1["losses"]) == num_mb
    assert all(l > 0 for l in r1["losses"])


# ══════════════════════════════════════════════════════════════
# TEST 5: Engine Stats
# ══════════════════════════════════════════════════════════════

def test_engine_stats():
    stats = EngineStats(
        execution_id="test",
        stage_index=0,
        num_microbatches=4,
        total_forwards=4,
        total_backwards=4,
        forward_time=2.0,
        backward_time=3.0,
        idle_time=1.0,
        start_time=100.0,
        end_time=106.0,
    )
    assert stats.total_time == 6.0
    # GPU util = (2+3)/6 = 0.833
    assert abs(stats.gpu_utilization - 0.833) < 0.01
    d = stats.to_dict()
    assert d["total_forwards"] == 4
    assert d["gpu_utilization"] > 0.8


def test_schedule_step_from_dict():
    d = {
        "step_index": 5,
        "stage_index": 2,
        "action": "forward",
        "microbatch_index": 3,
        "clock_tick": 7,
        "depends_on": 2,
    }
    step = ScheduleStep.from_dict(d)
    assert step.step_index == 5
    assert step.stage_index == 2
    assert step.action == "forward"
    assert step.microbatch_index == 3
    assert step.depends_on == 2


def test_microbatch_slot():
    slot = MicrobatchSlot(index=0)
    assert not slot.forward_done
    assert not slot.backward_done
    assert slot.loss is None
    slot.forward_done = True
    slot.loss = 2.5
    assert slot.forward_done
    assert slot.loss == 2.5


# ══════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sections = [
        ("TEST 1: Schedule Generation", [
            ("sched_single_stage", test_schedule_single_stage),
            ("sched_multi_first", test_schedule_multi_stage_first),
            ("sched_multi_last", test_schedule_multi_stage_last),
            ("sched_microbatch_indices", test_schedule_microbatch_indices),
        ]),
        ("TEST 2: Activation Serialization", [
            ("serialize_roundtrip", test_serialize_roundtrip),
            ("serialize_dtypes", test_serialize_different_dtypes),
        ]),
        ("TEST 3: Engine — Single Stage", [
            ("engine_single_stage", test_engine_single_stage),
            ("engine_grad_update", test_engine_single_stage_grad_update),
        ]),
        ("TEST 4: Engine — Two-Stage Pipeline", [
            ("engine_two_stage", test_engine_two_stage_pipeline),
        ]),
        ("TEST 5: Data Structures", [
            ("engine_stats", test_engine_stats),
            ("schedule_step_from_dict", test_schedule_step_from_dict),
            ("microbatch_slot", test_microbatch_slot),
        ]),
    ]

    for section_name, tests in sections:
        print(f"\n{'=' * 60}")
        print(f"{section_name}")
        print(f"{'=' * 60}")
        for name, fn in tests:
            run_test(name, fn)

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  PASSED: {passed}/{passed + failed}")
    print(f"  FAILED: {failed}/{passed + failed}")

    if errors:
        print(f"\n  ERRORS:")
        for name, err in errors:
            print(f"    {name}: {err}")

    sys.exit(1 if failed else 0)
