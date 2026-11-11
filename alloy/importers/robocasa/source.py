# SPDX-License-Identifier: Apache-2.0
"""Read RoboCasa pack zips, local or remote, without unpacking the whole pack.

A remote zip is read with HTTP range requests (the zip index sits at its end), so listing a 2 GB pack or pulling one
asset's files costs a few megabytes. An *asset* is any folder in the pack that holds a ``model.xml``; its id is the
folder path below the pack's top folder, e.g. ``glass_cup/GlassCup003`` or ``blenders/Blender001/Blender001_Lid``.

Example::

    >>> from alloy.importers.robocasa.source import PackReader
    >>> reader = PackReader.open("objects_lightwheel")            # doctest: +SKIP
    >>> reader.assets()[:2]                                        # doctest: +SKIP
    ['aluminum_foil/AluminumFoil001', 'aluminum_foil/AluminumFoil002']
    >>> reader.extract("glass_cup/GlassCup003", Path("work"))     # doctest: +SKIP
    PosixPath('work/glass_cup/GlassCup003')
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from alloy.importers.robocasa.packs import Pack, pack

log = logging.getLogger("alloy.importers.robocasa")

MODEL = "model.xml"
_TIMEOUT_S = 120


class _RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file through HTTP range requests."""

    def __init__(self, url: str) -> None:
        super().__init__()
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
                self.url = r.geturl()
                self.size = int(r.headers["Content-Range"].split("/")[-1])
        except (OSError, KeyError, ValueError) as exc:
            raise OSError(f"cannot open {url} with range requests ({exc}); download the zip and pass its path "
                          "instead") from exc
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf: bytearray | memoryview) -> int:
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
            data = r.read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_zip(location: str | Path) -> zipfile.ZipFile:
    """A zip from a local path or an http(s) URL (range reads, nothing unpacked).

    Example::

        >>> open_zip("objects_lightwheel.zip").namelist()[:1]      # doctest: +SKIP
        ['README.md']
    """
    loc = str(location)
    if loc.startswith(("http://", "https://")):
        return zipfile.ZipFile(io.BufferedReader(_RangeFile(loc), buffer_size=1 << 20))
    path = Path(loc)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such zip; pass a pack zip path or URL")
    return zipfile.ZipFile(path)


@dataclass
class PackReader:
    """The assets of one pack zip.

    Example::

        >>> r = PackReader.open("objects_lightwheel", "objects_lightwheel.zip")   # doctest: +SKIP
        >>> len(r.assets())                                                     # doctest: +SKIP
        968
    """

    pack: Pack
    zf: zipfile.ZipFile

    @classmethod
    def open(cls, key: str, location: str | Path | None = None) -> PackReader:
        """Open pack ``key`` from ``location`` (zip path or URL); default: the pack's download URL."""
        p = pack(key)
        return cls(p, open_zip(location or p.url))

    def _rel(self, name: str) -> PurePosixPath | None:
        parts = PurePosixPath(name).parts
        if not parts or parts[0] != self.pack.top:
            return None
        return PurePosixPath(*parts[1:])

    def assets(self) -> list[str]:
        """Sorted asset ids (folders with a model.xml), relative to the pack's top folder."""
        out = []
        for name in self.zf.namelist():
            rel = self._rel(name)
            if rel is not None and rel.name == MODEL and len(rel.parts) >= 2:
                out.append(str(rel.parent))
        return sorted(out)

    def files(self, asset: str) -> list[zipfile.ZipInfo]:
        """The asset's own files (not those of a nested asset folder that has its own model.xml)."""
        nested = {a for a in self.assets() if a != asset and a.startswith(asset + "/")}
        out = []
        for info in self.zf.infolist():
            rel = self._rel(info.filename)
            if info.is_dir() or rel is None or not str(rel).startswith(asset + "/"):
                continue
            if any(str(rel).startswith(n + "/") for n in nested):
                continue
            out.append(info)
        if not out:
            raise KeyError(f"asset {asset!r} not in pack {self.pack.key!r}; list ids with PackReader.assets()")
        return out

    def extract(self, asset: str, dest: Path) -> Path:
        """Write the asset's files under ``dest/<asset>`` and return that folder (existing files are replaced)."""
        root = dest / asset
        for info in self.files(asset):
            rel = PurePosixPath(str(self._rel(info.filename))).relative_to(asset)
            if ".." in rel.parts:
                raise ValueError(f"{info.filename}: path leaves the asset folder; refusing to extract")
            out = root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(self.zf.read(info))
        log.debug("extracted %s/%s -> %s", self.pack.key, asset, root)
        return root


def category_of(asset: str) -> str:
    """The category (first folder) of an asset id.

    Example::

        >>> category_of("glass_cup/GlassCup003")
        'glass_cup'
    """
    return PurePosixPath(asset).parts[0]


__all__ = ["MODEL", "PackReader", "category_of", "open_zip"]
