# SPDX-License-Identifier: Apache-2.0
"""Keeps the live layer's background CUDA calls out of the training thread's ``env.step``.

The engine captures each control period as a CUDA graph. While a capture is open, a CUDA call from any other thread
that is not capture-safe (an event synchronize, an allocation, NVENC's copies) invalidates it, and the step fails
("operation failed due to a previous error during capture"). So the training thread holds the gate for the length
of ``env.step``, and the sender and encoder threads hold it for each of their short CUDA calls. The training thread
announces itself first, so a background thread never takes the gate again while training is waiting: the most a
step can wait is one background call already in flight (well under a millisecond).

Example::

    with GATE.training():
        env.step(actions)
    with GATE.background():        # in a background thread
        event.synchronize()
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


class GpuGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._want = threading.Event()
        self.waits = 0
        self.wait_s = 0.0

    @contextmanager
    def training(self) -> Iterator[None]:
        self._want.set()
        if not self._lock.acquire(blocking=False):
            t0 = time.perf_counter()
            self._lock.acquire()
            self.waits += 1
            self.wait_s += time.perf_counter() - t0
        self._want.clear()
        try:
            yield
        finally:
            self._lock.release()

    @contextmanager
    def background(self) -> Iterator[None]:
        while True:
            while self._want.is_set():
                time.sleep(0.0005)
            self._lock.acquire()
            if not self._want.is_set():
                break
            self._lock.release()                 # training asked meanwhile: let it go first
        try:
            yield
        finally:
            self._lock.release()


GATE = GpuGate()
