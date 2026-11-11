# SPDX-License-Identifier: Apache-2.0
"""Denoiser weights: manifest, local cache and download.

    from alloy.camera.denoiser.weights import resolve_weights, list_weights
    list_weights()                       # {"v0_128": {...}, ...}
    path = resolve_weights("default")    # cached .pt path; downloads from $ALLOY_DENOISER_URL on first use

Names resolve through ``weights.json`` next to this file. A name is downloaded once from ``base_url`` (argument,
else env ``ALLOY_DENOISER_URL``, else the public folder :func:`default_url`: ``denoiser/`` of the Alloy library on the
Hugging Face Hub, no account needed): an ``https://`` / ``http://`` / ``file://`` URL, a local directory, or a
``gs://`` folder (needs the ``gcloud`` CLI). Files are verified against the manifest sha256 and size, then cached in
``cache_dir`` (argument, else ``$ALLOY_CACHE/denoiser``, else ``~/.cache/alloy/denoiser``).

Offline: a cached file needs no network. ``ALLOY_DENOISER_URL=""`` (or ``HF_HUB_OFFLINE=1``) turns downloads off. To
fill the cache ahead of time, run ``python -m alloy.camera.denoiser.weights`` on a machine with network (it prints
the files), or copy the ``.pt`` files into the cache folder.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

log = logging.getLogger(__name__)

MANIFEST_PATH = Path(__file__).with_name("weights.json")
URL_ENV = "ALLOY_DENOISER_URL"
CACHE_ENV = "ALLOY_CACHE"
HUB_REPO = "argonrobotics-ai/alloy-library"  # the public Hub dataset of the object library; weights in denoiser/
HUB_FOLDER = "denoiser"
_CHUNK = 1 << 20
_PLACEHOLDER_SHA = "0" * 64


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Read the weights manifest (default: the one shipped with the package).

    Example: ``load_manifest()["version"] == 1``.
    """
    with open(path or MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_weights(manifest: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return the manifest's weights: name -> {file, sha256, bytes, train_res, spp, notes}.

    Example: ``sorted(list_weights())`` gives the known names.
    """
    return dict((manifest or load_manifest())["weights"])


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file, read in chunks. Example: ``sha256_file("dn.pt")``."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_in_manifest(digest: str, manifest: dict[str, Any] | None = None) -> bool:
    """True if some manifest entry has this sha256 (placeholder zeros never count)."""
    if digest == _PLACEHOLDER_SHA:
        return False
    return any(w.get("sha256") == digest for w in list_weights(manifest).values())


def default_cache_dir() -> Path:
    """Cache folder: ``$ALLOY_CACHE/denoiser`` if set, else ``~/.cache/alloy/denoiser``."""
    root = os.environ.get(CACHE_ENV)
    return Path(root) / "denoiser" if root else Path.home() / ".cache" / "alloy" / "denoiser"


def default_url() -> str:
    """The public folder that holds the weights (``$HF_ENDPOINT`` replaces huggingface.co, for a mirror).

    Example: ``default_url()`` ends in ``"/datasets/argonrobotics-ai/alloy-library/resolve/main/denoiser"``.
    """
    endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/") or "https://huggingface.co"
    return f"{endpoint}/datasets/{HUB_REPO}/resolve/main/{HUB_FOLDER}"


def download_url(base_url: str | None = None) -> str | None:
    """Where weights are downloaded from: ``base_url``, else ``$ALLOY_DENOISER_URL`` (set but empty: None, downloads
    off), else :func:`default_url` (None with ``HF_HUB_OFFLINE=1``). Example: ``download_url("/data/weights")``."""
    if base_url:
        return base_url
    env = os.environ.get(URL_ENV)
    if env is not None:
        return env.strip() or None
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")
    return None if offline else default_url()


def _expand_default(spec: str, manifest: dict[str, Any]) -> str:
    """Map "default" / "default:<res>" to a manifest name."""
    defaults = manifest.get("default", {})
    res = spec.split(":", 1)[1] if ":" in spec else "128"
    if res not in defaults:
        raise ValueError(f"No default weights for resolution {res!r}; choose one of {sorted(defaults)} "
                         f"(e.g. 'default:{next(iter(defaults), '128')}') or pass a weights name or .pt path.")
    return defaults[res]


def resolve_weights(name_or_path: str | Path, *, cache_dir: str | Path | None = None,
                    base_url: str | None = None, manifest: dict[str, Any] | None = None) -> Path:
    """Return a local .pt path for a file path, "default[:128|:640x480]" or a manifest name.

    Existing files are returned as-is. Names are served from the cache when the sha256 matches, otherwise
    downloaded from :func:`download_url` (``base_url``, ``$ALLOY_DENOISER_URL``, else the public folder), verified
    and cached atomically.
    Example: ``resolve_weights("default")`` or ``resolve_weights("my_model.pt")``.
    """
    spec = str(name_or_path)
    if Path(spec).is_file():
        return Path(spec)
    manifest = manifest if manifest is not None else load_manifest()
    name = _expand_default(spec, manifest) if spec == "default" or spec.startswith("default:") else spec
    known = manifest["weights"]
    if name not in known:
        raise ValueError(f"Unknown denoiser weights {spec!r} (no such file either). Known names: "
                         f"{sorted(known)}. Pass one of them, 'default', or the path of an existing .pt file.")
    return _fetch_named(name, known[name], Path(cache_dir) if cache_dir else default_cache_dir(), base_url)


def _fetch_named(name: str, entry: dict[str, Any], cache: Path, base_url: str | None) -> Path:
    """Return the cached file for a manifest entry, downloading it first if needed."""
    target = cache / entry["file"]
    if target.is_file() and sha256_file(target) == entry["sha256"]:
        return target
    if entry["sha256"] == _PLACEHOLDER_SHA:
        raise ValueError(f"Weights {name!r} are a placeholder in the manifest (not released yet). "
                         f"Pass a local .pt path instead.")
    base = download_url(base_url)
    if not base:
        raise FileNotFoundError(f"Weights {name!r} are not cached in {cache} and downloads are off ({URL_ENV} is "
                                f"empty or HF_HUB_OFFLINE is set). Copy {entry['file']} into {cache}, unset them to "
                                f"download from {default_url()}, or pass a local .pt path.")
    log.info("downloading denoiser weights %s (%.0f MB) from %s", name, entry["bytes"] / 1e6, base)
    cache.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=cache, prefix=entry["file"] + ".", suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        _download(base, entry["file"], tmp)
        _verify(tmp, name, entry)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    log.info("denoiser weights %s cached at %s", name, target)
    return target


def _verify(path: Path, name: str, entry: dict[str, Any]) -> None:
    """Raise ValueError if the downloaded file's size or sha256 differs from the manifest."""
    size = path.stat().st_size
    if size != entry["bytes"]:
        raise ValueError(f"Downloaded {name!r} has {size} bytes, manifest says {entry['bytes']}. "
                         f"Check the folder behind {URL_ENV} holds the right file, then retry.")
    digest = sha256_file(path)
    if digest != entry["sha256"]:
        raise ValueError(f"sha256 mismatch for {name!r}: got {digest}, manifest says {entry['sha256']}. "
                         f"The file is corrupt or a different version; re-download or update the package.")


def _download(base: str, file: str, dst: Path) -> None:
    """Fetch ``<base>/<file>`` into dst. Supports https, http, file://, gs:// and local directories."""
    base = base.rstrip("/")
    scheme = urlparse(base).scheme
    if scheme in ("http", "https"):
        _download_http(f"{base}/{file}", dst)
    elif scheme == "gs":
        _download_gs(f"{base}/{file}", dst)
    elif scheme == "file":
        _copy_local(Path(url2pathname(urlparse(base).path)) / file, dst)
    elif scheme == "" or len(scheme) == 1:  # plain directory (or a Windows drive letter)
        _copy_local(Path(base) / file, dst)
    else:
        raise ValueError(f"Unsupported scheme {scheme!r} in {URL_ENV}={base!r}; "
                         f"use https://, http://, file://, gs:// or a local directory.")


def _download_http(url: str, dst: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "alloy-engine-denoiser"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(dst, "wb") as out:  # noqa: S310  (http/https only)
            shutil.copyfileobj(resp, out, _CHUNK)
    except urllib.error.HTTPError as e:
        file = url.rsplit("/", 1)[-1]
        what = "is not published there yet, or the address is wrong" if e.code == 404 else f"was refused ({e.reason})"
        raise FileNotFoundError(f"The denoiser weights file {file} {what}: {url} answered HTTP {e.code}. Check "
                                f"{URL_ENV} (unset: {default_url()}), update alloy-engine if the weights moved, or "
                                f"pass a local .pt path.") from e
    except OSError as e:
        raise ConnectionError(f"Could not reach {url} for the denoiser weights ({e}). They are fetched once and "
                              f"then cached: check the network (or {URL_ENV}), copy {url.rsplit('/', 1)[-1]} into "
                              f"{dst.parent} from another machine, or pass a local .pt path.") from e


def _download_gs(src: str, dst: Path) -> None:
    if shutil.which("gcloud") is None:
        raise RuntimeError("gs:// downloads need the gcloud CLI. Install it, or point "
                           f"{URL_ENV} at an https:// folder instead.")
    res = subprocess.run(["gcloud", "storage", "cp", src, str(dst)], check=False, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"gcloud storage cp {src} failed: {res.stderr.strip()}. "
                           f"Check access and the folder name, or pass a local .pt path.")


def _copy_local(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"{src} does not exist. Set {URL_ENV} to the folder that holds {src.name}, "
                                f"or pass a local .pt path.")
    shutil.copyfile(src, dst)


def main(argv: list[str] | None = None) -> int:
    """Fetch weights into the cache and print each file: ``python -m alloy.camera.denoiser.weights [NAME ...]``
    (no names: the defaults, "default" and "default:640x480"; "all": every manifest entry)."""
    names = list(sys.argv[1:] if argv is None else argv) or ["default", "default:640x480"]
    if names == ["all"]:
        names = sorted(list_weights())
    for name in names:
        try:
            print(f"{name}: {resolve_weights(name)}")  # noqa: T201  (a command's output)
        except (OSError, ValueError, RuntimeError) as e:
            print(f"{name}: {e}", file=sys.stderr)  # noqa: T201
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
