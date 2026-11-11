# SPDX-License-Identifier: Apache-2.0
"""alloy.live without a GPU: the wire format, commands and replay, settings, the GPU gate, frame packing, and world
forks on a small CPU scene (a fork is bit-identical at the fork; restore and pause are exact)."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from alloy.live.commands import CommandError, CommandLog, LogEntry, Replay, make_command
from alloy.live.frames import _flatten, inflate
from alloy.live.gate import GpuGate
from alloy.live.settings import SettingError, Settings
from alloy.live.wire import Kind, decode_body, encode, pack_arrays, unpack_arrays


def test_wire_roundtrip_with_arrays():
    arrays = {"reward": np.arange(64, dtype=np.float32), "flags": np.array([[1, 0], [0, 1]], np.uint8)}
    meta, blob = pack_arrays(arrays)
    frame = encode(Kind.NUMBERS, {"t": 1.5, **meta}, blob)
    kind, header, payload = decode_body(frame[4:])
    assert kind == Kind.NUMBERS and header["t"] == 1.5
    back = unpack_arrays(header, payload)
    for k, v in arrays.items():
        assert back[k].dtype == v.dtype and np.array_equal(back[k], v)


def test_commands_validate_and_replay_at_their_steps(tmp_path):
    with pytest.raises(CommandError):
        make_command("teleport", {})
    with pytest.raises(CommandError):
        make_command("push", {"world": 1})
    log = CommandLog(tmp_path / "commands.jsonl")
    for step, op, args in ((5, "set", {"name": "lr", "value": 1e-3}), (9, "reset", {"world": 2}),
                           (9, "pause", {"world": None})):
        c = make_command(op, args, issuer="agent:test")
        log.append(LogEntry(c["id"], op, args, "agent:test", step, time.time(), True, {}))
    replay = Replay(CommandLog.read(tmp_path / "commands.jsonl"))
    assert replay.due(4) == [] and [c["op"] for c in replay.due(5)] == ["set"]
    assert [c["op"] for c in replay.due(9)] == ["reset", "pause"] and replay.remaining == 0


def test_settings_whitelist_and_range():
    box = {"lr": 3e-4}
    s = Settings()
    s.add("lr", lambda: box["lr"], lambda v: box.__setitem__("lr", v), lo=1e-5, hi=1e-2)
    assert s.set("lr", 1e-3) == {"name": "lr", "old": 3e-4, "new": 1e-3}
    with pytest.raises(SettingError):
        s.set("lr", 1.0)
    with pytest.raises(SettingError):
        s.set("gravity", 0.0)


def test_gate_lets_training_in_first():
    gate, order = GpuGate(), []
    with gate.training():
        t = threading.Thread(target=lambda: (gate.background().__enter__(), order.append("bg")))
        t.start()
        time.sleep(0.05)
        order.append("train")                       # the background call waits for the step
    t.join(1.0)
    assert order == ["train", "bg"]


def test_frame_packing_roundtrip():
    frame = {"world": 3, "body_pos": np.ones((2, 3), np.float32), "meshes": {"sheet": np.zeros((4, 3), np.float32)},
             "topology": {"mesh_faces": {"sheet": np.array([[0, 1, 2]], np.int32)}, "body_names": ["a", "b"]}}
    arrays: dict[str, np.ndarray] = {}
    flat = _flatten(frame, "f0", arrays)
    meta, blob = pack_arrays(arrays)
    back = inflate(flat, unpack_arrays(meta, blob))
    assert back["body_pos"] == [[1.0] * 3] * 2 and back["topology"]["mesh_faces"]["sheet"] == [[0, 1, 2]]
    assert back["topology"]["body_names"] == ["a", "b"] and back["world"] == 3


@pytest.fixture(scope="module")
def env():
    from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task
    from alloy.envs.vector import VectorEnv

    e = VectorEnv(gripper_lift_scene(3), gripper_lift_task(), worlds=3, device="cpu", seed=0)
    e.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(12):
        e.step(e.xp.asarray(rng.uniform(-1, 1, (3, 3)).astype(np.float32)))
    yield e
    e.close()


def _rows(env, w):
    idx = env.sim.index
    rows = [t + w * idx.stride["body"] for t in sorted(idx.bodies.values())]
    return env.sim.engine.state.body_q.numpy()[rows].copy()


def test_fork_restore_pause_are_exact(env):
    from alloy.live.worldops import WorldOps

    ops = WorldOps(env)
    assert not np.array_equal(_rows(env, 0), _rows(env, 1))
    ops.fork(0, 1)
    assert np.array_equal(_rows(env, 0), _rows(env, 1))
    ops.save(0, "s")
    saved = _rows(env, 0)
    env.step(env.xp.asarray(np.ones((3, 3), np.float32)))
    assert not np.array_equal(_rows(env, 2), saved)
    ops.restore("s", 2)                                     # world 2 continues from world 0's saved moment
    assert np.array_equal(_rows(env, 2), saved)
    ops.pause(2)
    for _ in range(3):
        env.step(env.xp.asarray(np.ones((3, 3), np.float32)))
        ops.hold_frozen()
    assert np.array_equal(_rows(env, 2), saved)


def test_replaced_exporter_freed_in_capture_not_by_packer(monkeypatch):
    """A packer holds only a weak reference; a replaced exporter is dropped by capture() (the training thread)
    RETIRE_AFTER captures later, so its device buffers are never freed on the sender thread (WP13 review)."""
    import gc
    import threading
    import weakref

    import alloy.export
    from alloy.live import frames as F

    freed_on: list[str] = []

    class FakeExporter:
        def __init__(self, sim, worlds, max_points):
            self.worlds, self.refused, self.dropped = worlds, 0, 0

        def capture(self):
            return True

        def poll(self):
            return [{"world": self.worlds[0], "t": 0.0}]

        def __del__(self):
            freed_on.append(threading.current_thread().name)

    monkeypatch.setattr(alloy.export, "FrameExporter", FakeExporter)
    pf = F.PickedFrames(sim=None)
    pf.capture([0])
    first = weakref.ref(pf._exp)
    pack = pf.packer()
    pf.capture([1])                                   # new pick set: the first exporter is retired, not freed
    assert first() is not None and pack() is not None
    for _ in range(F.RETIRE_AFTER):
        pf.capture([1])
    gc.collect()
    assert first() is None and pack() is None         # freed by capture(); the old packer just yields nothing
    assert freed_on == [threading.current_thread().name]
