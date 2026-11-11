# SPDX-License-Identifier: Apache-2.0
"""The worker's link to the hub: one background thread dials out and sends, one reads commands; the training loop only
ever calls :meth:`Link.offer` (never blocks) and :meth:`Link.poll` (never blocks).

- Bounded outbox: when it is full the newest frame is dropped and counted (``dropped``), so a slow hub or network
  never slows training. Video frames are dropped as whole groups of pictures by the encoder (it restarts on a key
  frame), so a drop never corrupts the stream.
- Reconnects with backoff when the hub restarts; while disconnected, frames are dropped, commands simply wait.
- Frames whose payload is a callable are packed in the sender thread (the training thread only queues GPU->host
  copies; the waiting and packing happen here).

Example::

    link = Link("hub-host", 47100, hello={"worker": "vm1", "worlds": 64})
    link.offer(Kind.NUMBERS, {"t": 1.0}, payload_bytes)
    for kind, header, payload in link.poll():
        ...
"""
from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from alloy.live.wire import Kind, WireError, encode, read_frame

log = logging.getLogger(__name__)
Payload = bytes | Callable[[], tuple[dict[str, Any], bytes] | None]


class Link:
    """Dial-out connection to the hub with a bounded, drop-on-full outbox."""

    def __init__(self, host: str, port: int, hello: dict[str, Any], *, outbox: int = 256,
                 connect: Callable[[str, int], socket.socket] | None = None) -> None:
        self.host, self.port, self.hello = host, int(port), dict(hello)
        self._out: queue.Queue = queue.Queue(maxsize=outbox)
        self._in: queue.SimpleQueue = queue.SimpleQueue()
        self._connect = connect or (lambda h, p: socket.create_connection((h, p), timeout=5.0))
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self.connected = threading.Event()
        self.sent = self.dropped = self.sent_bytes = 0
        self.connections = 0                 # bumps on every (re)connect: per-connection frames are re-sent
        self._thread = threading.Thread(target=self._run, name="alloy-live-send", daemon=True)
        self._thread.start()

    # --- training-thread side (never blocks) ---------------------------------------------------------------------

    def offer(self, kind: Kind, header: dict[str, Any], payload: Payload = b"") -> bool:
        """Queue a frame; False (and counted in ``dropped``) when the outbox is full or the hub is not connected."""
        if not self.connected.is_set():
            self.dropped += 1
            return False
        try:
            self._out.put_nowait((kind, header, payload))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def poll(self) -> Iterator[tuple[Kind, dict[str, Any], bytes]]:
        """Frames received from the hub since the last poll (commands, picks)."""
        while True:
            try:
                yield self._in.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        self._stop.set()
        self._out.put(None) if not self._out.full() else None
        self._thread.join(timeout=2.0)
        self._drop_socket()

    # --- sender thread ---------------------------------------------------------------------------------------------

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            if self._sock is None and not self._dial():
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
                continue
            backoff = 0.5
            try:
                item = self._out.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            self._send(*item)

    def _dial(self) -> bool:
        try:
            sock = self._connect(self.host, self.port)
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(encode(Kind.HELLO, self.hello))
        except OSError as e:
            log.debug("live: hub %s:%d not reachable (%s)", self.host, self.port, e)
            return False
        self._sock = sock
        self.connections += 1
        threading.Thread(target=self._read, args=(sock,), name="alloy-live-recv", daemon=True).start()
        self.connected.set()
        log.info("live: connected to hub %s:%d as %s", self.host, self.port, self.hello.get("worker"))
        return True

    def _send(self, kind: Kind, header: dict[str, Any], payload: Payload) -> None:
        if callable(payload):
            try:
                packed = payload()
            except Exception:  # noqa: BLE001 - a bad sample must never stop the link
                log.exception("live: packing a %s frame failed; dropped", kind.name)
                self.dropped += 1
                return
            if packed is None:
                return
            extra, payload = packed
            header = {**header, **extra}
        try:
            frame = encode(kind, header, payload)
            self._sock.sendall(frame)
            self.sent += 1
            self.sent_bytes += len(frame)
        except (OSError, WireError, AttributeError) as e:
            log.warning("live: send failed (%s); reconnecting", e)
            self.dropped += 1
            self._drop_socket()

    def _read(self, sock: socket.socket) -> None:
        try:
            while True:
                msg = read_frame(sock)
                if msg is None:
                    break
                kind, header, payload = msg
                self._in.put((kind, header, bytes(payload)))
        except (OSError, WireError, ValueError) as e:
            log.debug("live: read stopped (%s)", e)
        if self._sock is sock:
            self._drop_socket()

    def _drop_socket(self) -> None:
        sock, self._sock = self._sock, None
        self.connected.clear()
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
