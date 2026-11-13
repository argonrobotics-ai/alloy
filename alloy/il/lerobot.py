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
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alloy.api.errors import FeatureUnavailableError
from alloy.envs.errors import EnvError
from alloy.il.episode import DatasetSpec, Episode

log = logging.getLogger("alloy.il")
DATASET_MODULE = "lerobot.datasets.lerobot_dataset"
IMAGE_NAMES = ["height", "width", "channels"]
LEROBOT_PYTHON = (3, 12)                                   # the lerobot releases this writer is tested with need it
DATASET_DIRS = {"meta", "data", "videos", "images"}       # what a LeRobot dataset directory holds at its top
ENV_STATE = "observation.environment_state"


def load_lerobot_dataset_class() -> Any:
    """The ``LeRobotDataset`` class, imported on first use; raises ``FeatureUnavailableError`` without lerobot.

    Example: ``load_lerobot_dataset_class().create(repo_id="local/x", root="out", fps=30, features={})``."""
    try:
        return importlib.import_module(DATASET_MODULE).LeRobotDataset
    except ImportError as e:
        if sys.version_info[:2] < LEROBOT_PYTHON:
            now = f"{sys.version_info[0]}.{sys.version_info[1]}"
            raise FeatureUnavailableError(
                f"LeRobot datasets need the lerobot package, which needs Python 3.12 (this is Python {now}, where "
                "the lerobot extra installs nothing)",
                "make a Python 3.12 environment and pip install 'alloy-engine[lerobot]' there; on this Python, "
                "record with HDF5Writer", field="lerobot") from e
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
    when it holds ``meta/info.json`` or nothing but a LeRobot dataset's own folders (what an interrupted
    recording leaves), never an arbitrary directory.

    Example: ``check_overwrite(Path("out/lift"), overwrite=True)``."""
    if not root.exists():
        return
    if not overwrite:
        raise EnvError(f"{root} already exists", "pass overwrite=True, or choose another root", field="root")
    entries = {p.name for p in root.iterdir()} if root.is_dir() else {root.name}
    partial = root.is_dir() and entries <= DATASET_DIRS and all((root / name).is_dir() for name in entries)
    if not (root / "meta" / "info.json").is_file() and not partial:
        raise EnvError(f"{root} exists and is not a LeRobot dataset (no meta/info.json, and it holds "
                       f"{sorted(entries - DATASET_DIRS)[:3]}), so it is not removed",
                       "choose an empty path, or delete the directory yourself", field="root")
    shutil.rmtree(root)


def unclosed_files(root: Path) -> list[Path]:
    """Parquet files under ``root`` without a footer: what a recording killed before ``close()`` leaves.

    Example: ``unclosed_files(Path("out/lift"))`` -> ``[]`` for a closed dataset."""
    out = []
    for path in sorted(root.rglob("*.parquet")):
        with open(path, "rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - 4))
            if size < 8 or f.read(4) != b"PAR1":
                out.append(path)
    return out


def feature_name(key: str) -> str:
    """An observation key as a LeRobot feature name part: LeRobot allows no ``/``, so ``gripper/wrist`` is
    stored as ``gripper_wrist``. Example: ``feature_name("gripper/wrist")``."""
    return key.replace("/", "_")


def has_state(spec: DatasetSpec) -> bool:
    """True when the main vector observation has columns (a camera-only task has none: no ``observation.state``).

    Example: ``has_state(spec)``."""
    shape = spec.obs_shapes.get(spec.state_key)
    return shape is not None and len(shape) == 1 and int(shape[0]) > 0


def vector_keys(spec: DatasetSpec) -> list[str]:
    """Observation keys stored as ``observation.<k>``: float vectors (1-D per step, at least one column), other
    than the main one. Example: ``vector_keys(spec)`` -> ``["wrist_force"]``."""
    return [k for k, shape in spec.obs_shapes.items() if k != spec.state_key and spec.obs_dtypes[k] == "float32"
            and len(shape) == 1 and int(shape[0]) > 0]


def stored_keys(spec: DatasetSpec) -> dict[str, str]:
    """Observation key -> LeRobot feature name, for every key the layout can hold; raises ``EnvError`` when two
    keys get one name. Example: ``stored_keys(spec)["state"]`` -> ``"observation.state"``."""
    keys = {spec.state_key: "observation.state"} if has_state(spec) else {}
    keys.update({k: f"observation.images.{feature_name(k)}" for k in spec.image_keys})
    keys.update({k: f"observation.{feature_name(k)}" for k in vector_keys(spec)})
    names = list(keys.values())
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise EnvError(f"two observation keys map to the LeRobot feature {twice[0]!r} (a '/' in a key is stored "
                       f"as '_'): {sorted(k for k, n in keys.items() if n == twice[0])}",
                       "rename one of them in Task(cameras={...}) or Task(obs={...})", field="obs")
    return keys


def split_state(spec: DatasetSpec, env_state: Sequence[str]) -> tuple[list[int], list[int]]:
    """Column indices of the main vector that stay in ``observation.state`` and those that go to
    ``observation.environment_state``: a column goes when its label starts with one of ``env_state``.

    Example: ``split_state(spec, ["cube"])``."""
    if not env_state:
        return list(range(len(spec.obs_names))), []
    if not has_state(spec) or not spec.obs_names:
        raise EnvError("env_state needs a labelled state vector, and this recording has none",
                       "leave env_state out for this env", field="env_state")
    for prefix in env_state:
        if not any(n.startswith(prefix) for n in spec.obs_names):
            raise EnvError(f"env_state prefix {prefix!r} matches no observation column",
                           f"use the start of labels from env.obs_names(), e.g. {spec.obs_names[:2]}",
                           field="env_state")
    moved = [i for i, n in enumerate(spec.obs_names) if any(n.startswith(p) for p in env_state)]
    kept = [i for i in range(len(spec.obs_names)) if i not in set(moved)]
    if not kept:
        raise EnvError("env_state takes every observation column, leaving observation.state empty",
                       "name the object columns only (the robot's stay in observation.state)", field="env_state")
    return kept, moved


def build_features(spec: DatasetSpec, video: bool, env_state: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """The LeRobot ``features`` dict for ``spec`` (see ``LeRobotWriter`` for the mapping).

    Example: ``build_features(spec, video=True)["observation.state"]["shape"]`` -> ``(6,)``."""
    f32 = "float32"
    feats: dict[str, dict[str, Any]] = {
        "action": {"dtype": f32, "shape": (spec.action_dim,), "names": list(spec.action_names)}}
    keys = stored_keys(spec)
    for k, name in keys.items():
        if k in spec.image_keys:
            feats[name] = {"dtype": "video" if video else "image", "shape": tuple(spec.obs_shapes[k]),
                           "names": list(IMAGE_NAMES)}
        else:
            labels = spec.obs_key_names.get(k) or (spec.obs_names if k == spec.state_key else None)
            labels = list(labels) if labels and len(labels) == int(spec.obs_shapes[k][0]) else None
            feats[name] = {"dtype": f32, "shape": tuple(spec.obs_shapes[k]), "names": labels}
    kept, moved = split_state(spec, env_state)
    if moved:
        feats["observation.state"] = {"dtype": f32, "shape": (len(kept),), "names": [spec.obs_names[i] for i in kept]}
        feats[ENV_STATE] = {"dtype": f32, "shape": (len(moved),), "names": [spec.obs_names[i] for i in moved]}
    for name, dtype in (("reward", f32), ("done", "bool"), ("success", "bool")):
        feats[f"next.{name}"] = {"dtype": dtype, "shape": (1,), "names": None}
    return feats


def frame_at(episode: Episode, t: int, spec: DatasetSpec, keys: dict[str, str],
             split: tuple[list[int], list[int]] | None = None) -> dict[str, Any]:
    """The LeRobot frame dict for step ``t``; ``keys`` maps observation key -> feature name, ``split`` is
    ``split_state``'s result when part of the state goes to ``observation.environment_state``.

    Example: ``dataset.add_frame(frame_at(ep, 0, spec, {"state": "observation.state"}))``."""
    last = t == episode.length - 1
    frame: dict[str, Any] = {"task": spec.task, "action": np.asarray(episode.actions[t], dtype=np.float32),
                             "next.reward": np.asarray([episode.rewards[t]], dtype=np.float32),
                             "next.done": np.asarray([episode.dones[t]], dtype=bool),
                             "next.success": np.asarray([last and episode.success], dtype=bool)}
    for k, name in keys.items():
        value = np.asarray(episode.obs[k][t])
        frame[name] = value if value.dtype == np.uint8 else value.astype(np.float32)
    if split is not None and split[1]:
        state = frame["observation.state"]
        frame["observation.state"], frame[ENV_STATE] = state[split[0]], state[split[1]]
    return frame


class LeRobotWriter:
    """A ``Writer`` that makes a LeRobotDataset v3.0 at ``root`` (local only; nothing touches the Hub).

    Feature mapping (per frame):

    - the state vector (``obs["state"]``, or the first group of ``Task(obs={...})``) -> ``observation.state``
      (float32, ``names`` are its column labels); a task with no state columns (cameras only) has no such feature
    - ``env_state=["cube"]`` moves the columns whose label starts with one of those names into
      ``observation.environment_state`` (LeRobot's ACT and Diffusion configs ask for it, or an image)
    - ``actions`` -> ``action`` (float32, ``names=spec.action_names``)
    - each camera key ``k`` (uint8 (H, W, 3)) -> ``observation.images.<k>``: an MP4 video when ``video=True``
      (lossy; needs PyAV, which ``lerobot[dataset]`` installs), else PNG images
    - any other float32 vector obs key ``k`` (another observation group) -> ``observation.<k>``
    - a ``/`` in a key is stored as ``_`` (``gripper/wrist`` -> ``observation.images.gripper_wrist``)
    - ``next.reward`` (float32), ``next.done`` and ``next.success`` (bool), each of shape (1,); ``next.success``
      is True only on the last frame of a successful episode
    - the task string of every frame is ``spec.task``

    Observation keys it cannot store (a float (H, W) depth map, say) are skipped with one warning each on the
    ``alloy.il`` logger. ``next_obs``, ``states`` and ``start`` are not part of the LeRobot layout and are not
    written (the HDF5 writer keeps them). ``fps`` must be a whole number. ``episodes`` and ``frames`` count what
    this writer wrote.

    The files are readable only after ``close()``: a process killed before it leaves a directory LeRobot cannot
    open (``overwrite=True`` replaces it). For a long run, record with ``HDF5Writer`` (readable after every
    episode, ``resume=True`` adds to it) and make the LeRobot dataset at the end with
    ``alloy.il.copy_hdf5(path, LeRobotWriter(root))``. ``resume=True`` appends to a closed dataset at ``root``
    (same features and fps) instead of refusing it.
    Example::

        with LeRobotWriter("out/lift", repo_id="local/lift", video=False) as writer:
            writer.open(spec)
            writer.write(episode)                       # episode.index is set
    """

    def __init__(self, root: str | Path, *, repo_id: str = "local/alloy", robot_type: str = "alloy",
                 video: bool = True, overwrite: bool = False, env_state: Sequence[str] = (),
                 resume: bool = False) -> None:
        if resume and overwrite:
            raise EnvError("resume and overwrite are both set", "pass one: resume=True appends, overwrite=True "
                           "starts again", field="resume")
        self.root, self.repo_id, self.robot_type = Path(root), repo_id, robot_type
        self.video, self.overwrite, self.resume = video, overwrite, resume
        self.env_state = tuple(env_state)
        self.episodes = 0
        self.frames = 0
        self.first_index = 0                                # episodes already in a resumed dataset
        self._dataset: Any = None
        self._spec: DatasetSpec | None = None
        self._keys: dict[str, str] = {}
        self._split: tuple[list[int], list[int]] | None = None

    def open(self, spec: DatasetSpec) -> None:
        """Create the dataset on disk for ``spec`` (or reopen it, with ``resume``). Example: ``writer.open(spec)``."""
        dataset_class = load_lerobot_dataset_class()
        fps = integer_fps(spec.fps)
        keys, split = self._stored_keys(spec), split_state(spec, self.env_state)
        features = build_features(spec, self.video, self.env_state)
        if self.resume and self.root.exists():
            self._dataset = self._reopen(dataset_class, features, fps)
            self.first_index = int(self._dataset.meta.total_episodes)
        else:
            check_overwrite(self.root, self.overwrite)
            self._dataset = dataset_class.create(repo_id=self.repo_id, root=self.root, fps=fps,
                                                 robot_type=self.robot_type, features=features,
                                                 use_videos=self.video)
        self._spec, self._keys, self._split = spec, keys, split

    def _reopen(self, dataset_class: Any, features: dict[str, Any], fps: int) -> Any:
        """The closed dataset at ``root``, open for more episodes, after checking it against ``features``."""
        fix = ("pass overwrite=True to start again there, or choose another root (for long runs record with "
               "HDF5Writer and convert at the end: alloy.il.copy_hdf5)")
        if not (self.root / "meta" / "info.json").is_file():
            raise EnvError(f"{self.root} holds no closed LeRobot dataset to resume (no meta/info.json)", fix,
                           field="root")
        open_files = unclosed_files(self.root)
        if open_files:
            raise EnvError(f"the LeRobot dataset at {self.root} was not closed ({open_files[0].name} has no parquet "
                           "footer): the recording that made it was interrupted before close()", fix, field="root")
        try:
            dataset = dataset_class.resume(repo_id=self.repo_id, root=self.root)
        except Exception as e:                              # lerobot's own errors, whatever their type
            raise EnvError(f"the LeRobot dataset at {self.root} cannot be reopened ({type(e).__name__}: "
                           f"{str(e).splitlines()[0][:120]})", fix, field="root") from e
        have = {k: tuple(v["shape"]) for k, v in dataset.meta.features.items() if k in features}
        want = {k: tuple(v["shape"]) for k, v in features.items()}
        if have != want or int(dataset.meta.fps) != fps:
            differ = sorted(k for k in want if have.get(k) != want[k])[:3]
            raise EnvError(f"the dataset at {self.root} does not match this recording (features that differ: "
                           f"{differ}; fps {dataset.meta.fps} against {fps})",
                           "resume with the env and writer settings that made it, or choose another root",
                           field="root")
        return dataset

    @staticmethod
    def _stored_keys(spec: DatasetSpec) -> dict[str, str]:
        keys = stored_keys(spec)
        for k in spec.obs_shapes:
            if k not in keys and not (k == spec.state_key and not has_state(spec)):
                log.warning("LeRobotWriter skips observation key %r (shape %s, %s): LeRobot stores state, camera "
                            "images and float vectors", k, tuple(spec.obs_shapes[k]), spec.obs_dtypes[k])
        return keys

    def write(self, episode: Episode) -> None:
        """Add one episode and set ``episode.index``. Example: ``writer.write(ep)``."""
        if self._dataset is None or self._spec is None:
            raise EnvError("the writer is not open", "call open(spec) first, or let a Recorder do it",
                           field="open")
        for t in range(episode.length):
            self._dataset.add_frame(frame_at(episode, t, self._spec, self._keys, self._split))
        self._dataset.save_episode()
        episode.index = self.first_index + self.episodes
        self.episodes += 1
        self.frames += episode.length

    def close(self) -> None:
        """Finish the dataset (``finalize`` writes the parquet footers). A dataset this writer made and wrote no
        episode to is removed: LeRobot cannot open an empty one. Safe to call twice.

        Example: ``writer.close()``."""
        if self._dataset is None:
            return
        self._dataset.finalize()
        self._dataset = None
        if self.episodes == 0 and self.first_index == 0:
            log.warning("LeRobotWriter: no episode was written, so the empty dataset at %s is removed (LeRobot "
                        "cannot open a dataset without episodes)", self.root)
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> LeRobotWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["LeRobotWriter", "build_features", "feature_name", "integer_fps", "load_lerobot_dataset_class",
           "split_state", "stored_keys", "unclosed_files"]
