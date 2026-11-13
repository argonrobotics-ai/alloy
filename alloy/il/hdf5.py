# SPDX-License-Identifier: Apache-2.0
"""Episodes in an HDF5 file with the robomimic dataset layout, so a recording trains with robomimic unchanged.

The file holds ``data/demo_<i>/{actions, rewards, dones, obs/<key>, next_obs/<key>, states}`` and the
``mask/{all, success}`` filter keys, as robomimic reads them; Alloy adds ``terminated``, ``truncated`` and
``start/<key>`` (what ``replay`` restores) to each demo, which robomimic ignores. Needs the ``h5py`` package
(``pip install h5py``). Example::

    from alloy.il.hdf5 import HDF5Writer, load_hdf5
    with HDF5Writer("lift.hdf5") as writer:        # a Recorder opens it with the dataset's DatasetSpec
        rec = Recorder(env, [writer])
    spec, episodes = load_hdf5("lift.hdf5")
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from alloy.api.errors import FeatureUnavailableError
from alloy.envs.errors import EnvError
from alloy.il.episode import DatasetSpec, Episode

log = logging.getLogger("alloy.il")

GYM_ENV_TYPE = 2          # robomimic's EnvType.GYM_TYPE: the value its env_args "type" holds for a gym-style env
DEMO = "demo_"


def _h5py() -> Any:
    """The ``h5py`` module, imported on first use. Example: ``_h5py().File(path, "r")``."""
    try:
        import h5py  # noqa: PLC0415
    except ImportError as e:
        raise FeatureUnavailableError(
            "HDF5 datasets need the h5py package",
            "pip install h5py   (in the environment Alloy is installed in)",
            field="h5py") from e
    return h5py


def _check_episode(spec: DatasetSpec, ep: Episode) -> None:
    """Raise ``EnvError`` when ``ep`` does not fit ``spec``. Example: ``_check_episode(spec, episode)``."""
    steps = ep.length
    for name, group in (("obs", ep.obs), ("next_obs", ep.next_obs)):
        if set(group) != set(spec.obs_shapes):
            raise EnvError(f"episode {name} keys {sorted(group)} differ from the dataset's "
                           f"{sorted(spec.obs_shapes)}",
                           "record every observation key of the DatasetSpec in every episode", field=name)
        for key, arr in group.items():
            if arr.shape != (steps, *spec.obs_shapes[key]) or arr.dtype != np.dtype(spec.obs_dtypes[key]):
                raise EnvError(f"{name}[{key!r}] is {arr.dtype}{arr.shape}, the dataset expects "
                               f"{spec.obs_dtypes[key]}{(steps, *spec.obs_shapes[key])}",
                               "make the episode's arrays match DatasetSpec.obs_shapes / obs_dtypes", field=name)
    if steps == 0 or ep.actions.shape != (steps, spec.action_dim):
        raise EnvError(f"actions are {ep.actions.shape}, the dataset expects (T >= 1, {spec.action_dim})",
                       "give every episode at least one step and one action column per spec.action_names",
                       field="actions")
    for name in ("rewards", "terminated", "truncated"):
        if getattr(ep, name).shape != (steps,):
            raise EnvError(f"{name} has shape {getattr(ep, name).shape}, expected ({steps},)",
                           "give rewards, terminated and truncated one value per step", field=name)
    if ep.states is not None and ep.states.shape[0] != steps:
        raise EnvError(f"states has {ep.states.shape[0]} rows for {steps} steps", "record one state per step",
                       field="states")


def _demo_number(name: str) -> int:
    """The number in a ``demo_<i>`` group name. Example: ``_demo_number("demo_12")`` is 12."""
    return int(name[len(DEMO):])


def _demo_names(data: Any) -> list[str]:
    """The demo groups of ``data`` in numeric order (demo_10 after demo_9). Example: ``_demo_names(f["data"])``."""
    return sorted((k for k in data if k.startswith(DEMO)), key=_demo_number)


def _read_group(group: Any) -> dict[str, np.ndarray]:
    """All datasets under ``group`` as ``{path: array}`` ("/" joins nested names). Example: ``_read_group(g)``."""
    out: dict[str, np.ndarray] = {}
    for key, item in group.items():
        if hasattr(item, "items"):
            out.update({f"{key}/{k}": v for k, v in _read_group(item).items()})
        else:
            out[key] = item[()]
    return out


class HDF5Writer:
    """Writes episodes into one robomimic-layout HDF5 file, kept readable after every ``write``.

    ``resume=True`` appends to a file an earlier run wrote (same observation and action layout; a file from a
    killed run is fine, it holds every episode written before). ``next_obs_images=False`` leaves the camera
    frames out of ``next_obs`` (they are ``obs`` shifted by one step plus one last frame), which halves the
    image data; robomimic algorithms that read ``next_obs`` images then need the default.

    Example: ``with HDF5Writer("lift.hdf5", overwrite=True) as w: w.open(spec); w.write(episode)``;
    ``w.episodes`` and ``w.samples`` count what the file holds."""

    def __init__(self, path: str | os.PathLike[str], *, compression: str | None = "gzip",
                 overwrite: bool = False, resume: bool = False, next_obs_images: bool = True) -> None:
        if resume and overwrite:
            raise EnvError("resume and overwrite are both set", "pass one: resume=True appends, overwrite=True "
                           "starts again", field="resume")
        self.path = os.fspath(path)
        self.compression = compression
        self.overwrite, self.resume, self.next_obs_images = overwrite, resume, next_obs_images
        self.episodes = 0
        self.samples = 0
        self.spec: DatasetSpec | None = None
        self._file: Any = None
        self._success: list[str] = []

    def open(self, spec: DatasetSpec) -> None:
        """Create the file and write the dataset attributes. Example: ``writer.open(spec)``."""
        h5py = _h5py()
        if os.path.exists(self.path) and self.resume:
            self._reopen(h5py, spec)
            return
        if os.path.exists(self.path) and not self.overwrite:
            raise EnvError(f"{self.path} already exists", "pass overwrite=True to replace it, resume=True to add "
                           "to it, or choose another path", field="path")
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.spec = spec
        self._file = h5py.File(self.path, "w")
        data = self._file.create_group("data")
        env_args = {"env_name": spec.env, "type": GYM_ENV_TYPE, "env_kwargs": spec.env_kwargs}
        data.attrs["env_args"] = json.dumps(env_args)
        data.attrs["alloy_spec"] = json.dumps(spec.to_json())
        data.attrs["total"] = 0
        self._write_masks([])
        self._file.flush()
        log.info("HDF5 dataset opened: path=%s", self.path)

    def _reopen(self, h5py: Any, spec: DatasetSpec) -> None:
        """Open the existing file for more episodes, after checking that it holds the same layout."""
        old, _ = load_hdf5(self.path, episodes=[])
        same = (old.obs_shapes == spec.obs_shapes and old.obs_dtypes == spec.obs_dtypes
                and old.action_names == spec.action_names and old.action_mode == spec.action_mode)
        if not same:
            raise EnvError(f"{self.path} holds another layout (observations {old.obs_shapes}, actions "
                           f"{old.action_names}) than this recording", "resume with the env that made it, or "
                           "choose another path", field="path")
        self.spec = spec
        self._file = h5py.File(self.path, "a")
        data = self._file["data"]
        demos = sorted(int(k[len(DEMO):]) for k in data if k.startswith(DEMO))
        self.episodes, self.samples = (demos[-1] + 1 if demos else 0), int(data.attrs.get("total", 0))
        self._success = [f"{DEMO}{i}" for i in demos if bool(data[f"{DEMO}{i}"].attrs.get("success", False))]
        log.info("HDF5 dataset reopened: path=%s episodes=%d", self.path, self.episodes)

    def write(self, episode: Episode) -> None:
        """Append ``episode`` as ``demo_<n>`` and set ``episode.index``. Example: ``writer.write(episode)``."""
        if self._file is None or self.spec is None:
            raise EnvError("write() was called before open() or after close()",
                           "call writer.open(spec) first (a Recorder does), and write before close()",
                           field="writer")
        _check_episode(self.spec, episode)
        episode.index = self.episodes
        self._write_demo(self._file["data"].create_group(f"{DEMO}{episode.index}"), episode)
        self.episodes += 1
        self.samples += episode.length
        if episode.success:
            self._success.append(f"{DEMO}{episode.index}")
        self._file["data"].attrs["total"] = self.samples
        self._write_masks(self._success)
        self._file.flush()

    def close(self) -> None:
        """Flush and close the file; safe to call twice. Example: ``writer.close()``."""
        if self._file is None:
            return
        self._file.close()
        self._file = None
        log.info("HDF5 dataset closed: path=%s episodes=%d samples=%d", self.path, self.episodes, self.samples)

    def __enter__(self) -> HDF5Writer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _write_masks(self, success: list[str]) -> None:
        """Rewrite the ``mask/all`` and ``mask/success`` filter keys (bytes arrays, as robomimic stores them)."""
        mask = self._file.require_group("mask")
        for name, demos in (("all", [f"{DEMO}{i}" for i in range(self.episodes)]), ("success", success)):
            if name in mask:
                del mask[name]
            mask.create_dataset(name, data=np.array(demos, dtype="S"))

    def _write_demo(self, demo: Any, ep: Episode) -> None:
        """Write one episode's datasets and attributes into the new ``demo`` group."""
        demo.attrs.update({"num_samples": ep.length, "success": bool(ep.success), "world": int(ep.world),
                           "episode_return": ep.episode_return, "invalid": bool(ep.invalid)})
        demo.create_dataset("actions", data=ep.actions.astype(np.float32))
        demo.create_dataset("rewards", data=ep.rewards.astype(np.float32))
        demo.create_dataset("dones", data=ep.dones.astype(np.int64))
        demo.create_dataset("terminated", data=ep.terminated.astype(bool))
        demo.create_dataset("truncated", data=ep.truncated.astype(bool))
        if ep.states is not None:
            demo.create_dataset("states", data=ep.states.astype(np.float32))
        for group, arrays in (("obs", ep.obs), ("next_obs", ep.next_obs)):
            for key, arr in arrays.items():
                image = key in self.spec.image_keys
                if image and group == "next_obs" and not self.next_obs_images:
                    continue
                self._write_obs(demo, f"{group}/{key}", arr, image)
        for key, arr in ep.start.items():
            demo.create_dataset(f"start/{key}", data=np.asarray(arr))

    def _write_obs(self, demo: Any, path: str, arr: np.ndarray, is_image: bool) -> None:
        """One observation dataset; images are compressed, vectors are not."""
        demo.create_dataset(path, data=arr, compression=self.compression if is_image else None)


def load_hdf5(path: str | os.PathLike[str], episodes: Sequence[int] | None = None
              ) -> tuple[DatasetSpec, list[Episode]]:
    """Read a file written by ``HDF5Writer`` back as its spec and episodes (all, or the demo numbers listed).

    Example: ``spec, eps = load_hdf5("lift.hdf5", episodes=[0, 3])``."""
    h5py = _h5py()
    with h5py.File(os.fspath(path), "r") as f:
        if "data" not in f or "alloy_spec" not in f["data"].attrs:
            raise EnvError(f"{path} was not written by Alloy (no data/alloy_spec attribute)",
                           "load files recorded with alloy.il.HDF5Writer; for other robomimic files use robomimic",
                           field="path")
        spec = DatasetSpec.from_json(json.loads(f["data"].attrs["alloy_spec"]))
        wanted = None if episodes is None else {int(i) for i in episodes}
        names = [n for n in _demo_names(f["data"]) if wanted is None or _demo_number(n) in wanted]
        return spec, [_read_episode(f["data"][n], _demo_number(n)) for n in names]


def _read_episode(demo: Any, index: int) -> Episode:
    """Rebuild the ``Episode`` stored in the ``demo`` group. Example: ``_read_episode(f["data/demo_0"], 0)``."""
    return Episode(
        obs=_read_group(demo["obs"]), next_obs=_read_group(demo["next_obs"]) if "next_obs" in demo else {},
        actions=demo["actions"][()],
        rewards=demo["rewards"][()], terminated=demo["terminated"][()].astype(bool),
        truncated=demo["truncated"][()].astype(bool), success=bool(demo.attrs["success"]),
        states=demo["states"][()] if "states" in demo else None,
        start=_read_group(demo["start"]) if "start" in demo else {},
        world=int(demo.attrs["world"]), index=index, invalid=bool(demo.attrs.get("invalid", False)))


def copy_hdf5(path: str | os.PathLike[str], writer: Any, episodes: Sequence[int] | None = None) -> int:
    """Write the episodes of an Alloy HDF5 file (all, or the demo numbers listed) to another ``writer``, one at a
    time, then close the writer; returns how many. A long recording can go to ``HDF5Writer`` (readable after every
    episode, resumable) and become a LeRobot dataset at the end.

    Example: ``copy_hdf5("lift.hdf5", LeRobotWriter("out/lift", repo_id="local/lift"))``."""
    h5py = _h5py()
    spec, _ = load_hdf5(path, episodes=[])
    wanted = None if episodes is None else {int(i) for i in episodes}
    writer.open(spec)
    count = 0
    try:
        with h5py.File(os.fspath(path), "r") as f:
            for name in _demo_names(f["data"]):
                if wanted is None or _demo_number(name) in wanted:
                    writer.write(_read_episode(f["data"][name], _demo_number(name)))
                    count += 1
    finally:
        writer.close()
    return count


__all__ = ["HDF5Writer", "copy_hdf5", "load_hdf5"]
