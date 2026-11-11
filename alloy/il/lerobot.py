# SPDX-License-Identifier: Apache-2.0
"""Write recorded episodes as a LeRobotDataset v3.0, through the ``lerobot`` package's own API.

``lerobot`` is optional: the ``lerobot`` extra (``pip install 'alloy-engine[lerobot]'``, Python 3.12) or
``pip install 'lerobot[dataset]'`` (tested with lerobot 0.7.0), in the same environment as Alloy. Importing this
module does not need it. Example::

    from alloy.il.lerobot import LeRobotWriter
    with LeRobotWriter("out/lift", repo_id="local/lift") as writer:     # a Recorder's writer
        rec = Recorder(env, [writer])
"""

from __future__ import annotations

import importlib
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from alloy.api.errors import FeatureUnavailableError
from alloy.envs.errors import EnvError
from alloy.il.episode import DatasetSpec, Episode

log = logging.getLogger("alloy.il")
DATASET_MODULE = "lerobot.datasets.lerobot_dataset"
IMAGE_NAMES = ["height", "width", "channels"]


def load_lerobot_dataset_class() -> Any:
    """The ``LeRobotDataset`` class, imported on first use; raises ``FeatureUnavailableError`` without lerobot.

    Example: ``load_lerobot_dataset_class().create(repo_id="local/x", root="out", fps=30, features={})``."""
    try:
        return importlib.import_module(DATASET_MODULE).LeRobotDataset
    except ImportError as e:
        raise FeatureUnavailableError("LeRobot datasets need the lerobot package",
                                      "pip install 'lerobot[dataset]' (tested with lerobot 0.7.0), in the same "
                                      "environment as Alloy", field="lerobot") from e


def integer_fps(fps: float) -> int:
    """``fps`` as an int; raises ``EnvError`` when it is not within 1e-6 of a whole number (LeRobot stores an int).

    Example: ``integer_fps(100.0)`` -> 100; ``integer_fps(1 / 0.03)`` raises (use dt = 1/30 or 0.01)."""
    if abs(fps - round(fps)) > 1e-6:
        raise EnvError(f"LeRobot stores an integer fps, but this recording runs at {fps:.6g} steps per second",
                       f"use a control period dt whose inverse is a whole number, e.g. dt=0.01 (100) or "
                       f"dt=1/30 (30); now dt={1 / fps:.6g}", field="fps")
    return int(round(fps))


def check_overwrite(root: Path, overwrite: bool) -> None:
    """Prepare ``root`` for a new dataset: refuse an existing one unless ``overwrite``, and then remove it only
    when it holds ``meta/info.json`` (never an arbitrary directory).

    Example: ``check_overwrite(Path("out/lift"), overwrite=True)``."""
    if not root.exists():
        return
    if not overwrite:
        raise EnvError(f"{root} already exists", "pass overwrite=True, or choose another root", field="root")
    if not (root / "meta" / "info.json").is_file():
        raise EnvError(f"{root} exists and is not a LeRobot dataset (no meta/info.json), so it is not removed",
                       "choose an empty path, or delete the directory yourself", field="root")
    shutil.rmtree(root)


def vector_keys(spec: DatasetSpec) -> list[str]:
    """Observation keys stored as ``observation.<k>``: float vectors (1-D per step), other than ``state``.

    Example: ``vector_keys(spec)`` -> ``["wrist_force"]``."""
    return [k for k, shape in spec.obs_shapes.items()
            if k != "state" and spec.obs_dtypes[k] == "float32" and len(shape) == 1]


def build_features(spec: DatasetSpec, video: bool) -> dict[str, dict[str, Any]]:
    """The LeRobot ``features`` dict for ``spec`` (see ``LeRobotWriter`` for the mapping).

    Example: ``build_features(spec, video=True)["observation.state"]["shape"]`` -> ``(6,)``."""
    f32 = "float32"
    feats: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": f32, "shape": tuple(spec.obs_shapes["state"]),
                              "names": list(spec.obs_names)},
        "action": {"dtype": f32, "shape": (spec.action_dim,), "names": list(spec.action_names)}}
    for k in spec.image_keys:
        feats[f"observation.images.{k}"] = {"dtype": "video" if video else "image",
                                            "shape": tuple(spec.obs_shapes[k]), "names": list(IMAGE_NAMES)}
    for k in vector_keys(spec):
        feats[f"observation.{k}"] = {"dtype": f32, "shape": tuple(spec.obs_shapes[k]), "names": None}
    for name, dtype in (("reward", f32), ("done", "bool"), ("success", "bool")):
        feats[f"next.{name}"] = {"dtype": dtype, "shape": (1,), "names": None}
    return feats


def frame_at(episode: Episode, t: int, spec: DatasetSpec, keys: dict[str, str]) -> dict[str, Any]:
    """The LeRobot frame dict for step ``t``; ``keys`` maps observation key -> feature name.

    Example: ``dataset.add_frame(frame_at(ep, 0, spec, {"state": "observation.state"}))``."""
    last = t == episode.length - 1
    frame: dict[str, Any] = {"task": spec.task, "action": np.asarray(episode.actions[t], dtype=np.float32),
                             "next.reward": np.asarray([episode.rewards[t]], dtype=np.float32),
                             "next.done": np.asarray([episode.dones[t]], dtype=bool),
                             "next.success": np.asarray([last and episode.success], dtype=bool)}
    for k, name in keys.items():
        value = np.asarray(episode.obs[k][t])
        frame[name] = value if value.dtype == np.uint8 else value.astype(np.float32)
    return frame


class LeRobotWriter:
    """A ``Writer`` that makes a LeRobotDataset v3.0 at ``root`` (local only; nothing touches the Hub).

    Feature mapping (per frame):

    - ``obs["state"]`` -> ``observation.state`` (float32, ``names=spec.obs_names``)
    - ``actions`` -> ``action`` (float32, ``names=spec.action_names``)
    - each camera key ``k`` (uint8 (H, W, 3)) -> ``observation.images.<k>``: an MP4 video when ``video=True``
      (lossy; needs PyAV, which ``lerobot[dataset]`` installs), else PNG images
    - any other float32 vector obs key ``k`` -> ``observation.<k>``
    - ``next.reward`` (float32), ``next.done`` and ``next.success`` (bool), each of shape (1,); ``next.success``
      is True only on the last frame of a successful episode
    - the task string of every frame is ``spec.task``

    Observation keys it cannot store (a float (H, W) depth map, say) are skipped with one warning each on the
    ``alloy.il`` logger. ``next_obs``, ``states`` and ``start`` are not part of the LeRobot layout and are not
    written (the HDF5 writer keeps them). ``fps`` must be a whole number. ``episodes`` and ``frames`` count what
    was written. Example::

        with LeRobotWriter("out/lift", repo_id="local/lift", video=False) as writer:
            writer.open(spec)
            writer.write(episode)                       # episode.index is set
    """

    def __init__(self, root: str | Path, *, repo_id: str = "local/alloy", robot_type: str = "alloy",
                 video: bool = True, overwrite: bool = False) -> None:
        self.root, self.repo_id, self.robot_type = Path(root), repo_id, robot_type
        self.video, self.overwrite = video, overwrite
        self.episodes = 0
        self.frames = 0
        self._dataset: Any = None
        self._spec: DatasetSpec | None = None
        self._keys: dict[str, str] = {}

    def open(self, spec: DatasetSpec) -> None:
        """Create the dataset on disk for ``spec``. Example: ``writer.open(spec)``."""
        dataset_class = load_lerobot_dataset_class()
        fps = integer_fps(spec.fps)
        check_overwrite(self.root, self.overwrite)
        self._spec, self._keys = spec, self._stored_keys(spec)
        self._dataset = dataset_class.create(repo_id=self.repo_id, root=self.root, fps=fps,
                                             robot_type=self.robot_type, features=build_features(spec, self.video),
                                             use_videos=self.video)

    @staticmethod
    def _stored_keys(spec: DatasetSpec) -> dict[str, str]:
        keys = {"state": "observation.state"}
        keys.update({k: f"observation.images.{k}" for k in spec.image_keys})
        keys.update({k: f"observation.{k}" for k in vector_keys(spec)})
        for k in spec.obs_shapes:
            if k not in keys:
                log.warning("LeRobotWriter skips observation key %r (shape %s, %s): LeRobot stores state, camera "
                            "images and float vectors", k, tuple(spec.obs_shapes[k]), spec.obs_dtypes[k])
        return keys

    def write(self, episode: Episode) -> None:
        """Add one episode and set ``episode.index``. Example: ``writer.write(ep)``."""
        if self._dataset is None or self._spec is None:
            raise EnvError("the writer is not open", "call open(spec) first, or let a Recorder do it",
                           field="open")
        for t in range(episode.length):
            self._dataset.add_frame(frame_at(episode, t, self._spec, self._keys))
        self._dataset.save_episode()
        episode.index = self.episodes
        self.episodes += 1
        self.frames += episode.length

    def close(self) -> None:
        """Finish the dataset (``finalize`` writes the parquet footers). Safe to call twice.

        Example: ``writer.close()``."""
        if self._dataset is not None:
            self._dataset.finalize()
            self._dataset = None

    def __enter__(self) -> LeRobotWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["LeRobotWriter", "build_features", "integer_fps", "load_lerobot_dataset_class"]
