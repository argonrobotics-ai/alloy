# SPDX-License-Identifier: Apache-2.0
"""Read a recording from a CSV file, an HDF5 file or a LeRobot dataset folder (``load_recording`` picks by path).

- **CSV**: ``alloy.sysid.recording`` (columns ``t, cmd.<joint>, pos.<joint>, ...``).
- **HDF5** (``pip install h5py``): datasets ``t`` (T,), ``command`` and ``position`` (T, J), optional ``velocity``
  and ``torque``, and the joint names in the attribute ``joints`` of the file or of each group; either at the top
  of the file (one episode) or one group per episode (any group names; read in sorted order).
- **LeRobot dataset** (a folder with ``meta/info.json``; ``pip install pyarrow``): ``action`` is read as the
  commanded joint targets and ``observation.state`` as the measured joint positions, ``timestamp`` as time, one
  episode per ``episode_index``. Joint names come from the features' ``names`` (pass ``joints=`` to name the
  columns yourself; ``columns=`` picks state columns when the state holds more than joint positions). The dataset
  must hold absolute joint targets: a dataset of delta or end-effector actions cannot be used, and Alloy's own
  ``alloy.il`` recordings hold delta actions unless the env's action mode is ``"absolute"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from alloy.sysid.errors import SysidError, need
from alloy.sysid.recording import Recording, load_csv, make_episode

HDF5_SUFFIXES = (".h5", ".hdf5", ".hdf")


def load_recording(path: str | Path, joints: list[str] | None = None, columns: list[int] | None = None,
                   aliases: dict[str, str] | None = None) -> Recording:
    """Read ``path`` by its kind. ``joints``, ``columns`` and ``aliases`` are ``load_lerobot``'s.
    Example: ``load_recording("take.csv")``, ``load_recording("my_dataset/")``."""
    path = Path(str(path)).expanduser()
    if path.is_dir():
        return load_lerobot(path, joints, columns, aliases)
    if not path.is_file():
        raise SysidError(f"recording {str(path)!r} does not exist", "pass a CSV or HDF5 file or a LeRobot dataset "
                         "folder", field="data")
    if path.suffix.lower() in HDF5_SUFFIXES:
        return load_hdf5(path)
    if path.suffix.lower() in (".csv", ".txt"):
        return load_csv(path)
    raise SysidError(f"{path.name}: unknown recording type {path.suffix!r}", "use .csv, .hdf5 or a LeRobot dataset "
                     "folder (docs/guide/sysid.md shows the layouts)", field="data")


def load_hdf5(path: str | Path) -> Recording:
    """Read the HDF5 layout of the module docstring. Example: ``load_hdf5("take.hdf5").joints``."""
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:
        raise need("h5py", "pip install h5py", "reading an HDF5 recording") from exc
    path = Path(path)
    with h5py.File(path, "r") as fh:
        groups = [fh] if "position" in fh else [fh[k] for k in sorted(fh) if "position" in fh[k]]
        if not groups:
            raise SysidError(f"{path.name}: no `position` dataset at the top or in a group", "store t, command, "
                             "position (and the attribute joints) per episode", field="data")
        names = None
        episodes = []
        for g in groups:
            missing = [k for k in ("t", "command") if k not in g]
            if missing:
                raise SysidError(f"{path.name}: {g.name} has no `{missing[0]}` dataset", "store t (T,), command "
                                 "(T, J) and position (T, J)", field="data")
            names = names or _names(g.attrs.get("joints", fh.attrs.get("joints")))
            episodes.append(make_episode(g["t"][()], g["command"][()], g["position"][()],
                                         g["velocity"][()] if "velocity" in g else None,
                                         g["torque"][()] if "torque" in g else None))
    count = episodes[0].position.shape[1]
    if names is None or len(names) != count:
        raise SysidError(f"{path.name}: the attribute `joints` must list the {count} joint names", "set "
                         "file.attrs['joints'] = ['joint1', ...] in the positions' column order", field="data")
    return Recording(names, episodes, source=str(path))


def save_hdf5(rec: Recording, path: str | Path) -> Path:
    """Write ``rec`` in the HDF5 layout (one group per episode). Example: ``save_hdf5(rec, "take.hdf5")``."""
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:
        raise need("h5py", "pip install h5py", "writing an HDF5 recording") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fh:
        fh.attrs["joints"] = list(rec.joints)
        for k, e in enumerate(rec.episodes):
            g = fh.create_group(f"episode_{k:04d}")
            for name, arr in (("t", e.t), ("command", e.command), ("position", e.position),
                              ("velocity", e.velocity), ("torque", e.torque)):
                if arr is not None:
                    g.create_dataset(name, data=arr)
    return path


def _names(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(raw).reshape(-1).tolist()]


def load_lerobot(folder: str | Path, joints: list[str] | None = None, columns: list[int] | None = None,
                 aliases: dict[str, str] | None = None) -> Recording:
    """Read a LeRobot dataset folder (module docstring). ``joints``: the joint of each ``action`` column (default:
    the dataset's action names, an actuator name replaced by its joint through ``aliases``); ``columns``: the
    ``observation.state`` column of each joint (default: the state columns named after the joints, as an
    ``alloy.il`` recording's ``<joint>.q``). Example: ``load_lerobot("data/so100_take")``."""
    folder = Path(folder)
    info_path = folder / "meta" / "info.json"
    if not info_path.is_file():
        raise SysidError(f"{folder.name}/ is not a LeRobot dataset (no meta/info.json)", "pass the dataset's root "
                         "folder, or a CSV / HDF5 file", field="data")
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:
        raise need("pyarrow", "pip install pyarrow", "reading a LeRobot dataset") from exc
    info = json.loads(info_path.read_text())
    files = sorted((folder / "data").rglob("*.parquet"))
    if not files:
        raise SysidError(f"{folder.name}/data holds no parquet files", "pass a LeRobot dataset with its data/ folder",
                         field="data")
    wanted = ["action", "observation.state", "timestamp", "episode_index"]
    try:
        tables = [pq.read_table(f, columns=wanted) for f in files]
    except Exception as exc:        # noqa: BLE001  pyarrow raises its own error types for a missing column
        raise SysidError(f"{folder.name}: its frames lack one of {wanted} ({exc})", "sysid needs `action` (joint "
                         "targets), `observation.state` (joint positions), `timestamp` and `episode_index`",
                         field="data") from exc
    col = {k: np.concatenate([np.asarray(t.column(k).to_pylist(), np.float64) for t in tables]) for k in wanted}
    action, state = col["action"].reshape(len(col["timestamp"]), -1), col["observation.state"].reshape(
        len(col["timestamp"]), -1)
    names = joints or _feature_names(info, "action") or _feature_names(info, "observation.state")
    names = None if names is None else [(aliases or {}).get(_strip(n), _strip(n)) for n in names]
    if columns is None:
        columns = _named_columns(names, _feature_names(info, "observation.state"), state.shape[1])
    if columns is not None:
        state = state[:, list(columns)]
    if action.shape[1] != state.shape[1] or names is None or len(names) != action.shape[1]:
        raise SysidError(f"{folder.name}: `action` has {action.shape[1]} columns, `observation.state` "
                         f"{state.shape[1]}, and {0 if names is None else len(names)} joint names are known",
                         "sysid needs one joint target and one joint position per joint. Name the joint of each "
                         "action column and the state column that holds its position: alloy.sysid.fit(robot, data, "
                         "column_names=[...], state_columns=[...]); command line --column-names a,b "
                         "--state-columns 0,2; load_recording(data, joints=[...], columns=[...])", field="data")
    episodes = [make_episode(col["timestamp"][m], action[m], state[m])
                for m in (col["episode_index"] == e for e in np.unique(col["episode_index"]))]
    rec = Recording(list(names), episodes, source=str(folder))
    rec.notes.append("LeRobot dataset: `action` read as absolute joint targets, `observation.state` as positions")
    return rec


def _feature_names(info: dict[str, Any], key: str) -> list[str] | None:
    names = info.get("features", {}).get(key, {}).get("names")
    if isinstance(names, dict):                       # some datasets nest them: {"motors": [...]}
        names = next(iter(names.values()), None)
    return [str(n) for n in names] if isinstance(names, (list, tuple)) and names else None


def _named_columns(joints: list[str] | None, state_names: list[str] | None, width: int) -> list[int] | None:
    """The state column named after each joint (``"arm/elbow.q"`` or ``"elbow.pos"`` for ``"elbow"``), or None
    when the state is not labelled per column or a joint has no column of its name."""
    if joints is None or state_names is None or len(state_names) != width:
        return None
    position = [None if n.endswith((".qd", ".vel")) else _strip(n) for n in state_names]
    if any(position.count(j) != 1 for j in joints):
        return None
    return [position.index(j) for j in joints]


def _strip(name: str) -> str:
    """A joint name without a ``.pos`` / ``.q`` suffix or a ``robot/`` prefix (``"arm/pan.pos"`` -> ``"pan"``)."""
    name = name.rsplit("/", 1)[-1]
    for suffix in (".pos", ".q"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


__all__ = ["load_hdf5", "load_lerobot", "load_recording", "save_hdf5"]
