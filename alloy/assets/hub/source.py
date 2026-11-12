# SPDX-License-Identifier: Apache-2.0
"""Access to the object library dataset on the Hugging Face Hub: repo id, token, index, file downloads, clear errors.

The repo id is ``$ALLOY_ASSETS_REPO`` when set, else ``DEFAULT_REPO``, which is public: no token is needed. When there
is one, huggingface_hub uses its own (``HF_TOKEN`` or the login saved by ``hf auth login``): it raises the Hub's rate
limit, and a private copy of the library needs it. Nothing here prints or stores a token.

Example::

    >>> repo_id()                                            # doctest: +SKIP
    'argonrobotics-ai/alloy-library'
    >>> rows = load_index()                                  # doctest: +SKIP
    >>> rows[0]["name"], rows[0]["library"]                  # doctest: +SKIP
    ('gar_barcode_scanning_bag_tote', 'gymanything')
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from alloy.api.errors import AlloyAPIError
from alloy.assets.hub.layout import INDEX

INDEX_CHECKED = ".index-checked"                         # touched when the index was last checked

log = logging.getLogger("alloy.assets.hub")

DEFAULT_REPO = "argonrobotics-ai/alloy-library"
REPO_ENV = "ALLOY_ASSETS_REPO"
REVISION_ENV = "ALLOY_ASSETS_REVISION"
REPO_TYPE = "dataset"
ACCESS_FIX = (f"check the dataset id ({DEFAULT_REPO} is public and needs no login); for a private copy "
              f"({REPO_ENV}=<owner>/<name>) request access on huggingface.co, then run `hf auth login` (or set "
              "HF_TOKEN); a stale saved token is refused too: run `hf auth logout` or unset HF_TOKEN")


class HubError(AlloyAPIError):
    """The object library could not be read from the Hub: ``message`` says what failed, ``fix`` what to do."""


def repo_id() -> str:
    """``$ALLOY_ASSETS_REPO`` or ``DEFAULT_REPO``. Example: ``repo_id()``."""
    return os.environ.get(REPO_ENV, "").strip() or DEFAULT_REPO


def revision() -> str | None:
    """``$ALLOY_ASSETS_REVISION`` (a branch, tag or commit), or None for the default branch."""
    return os.environ.get(REVISION_ENV, "").strip() or None


def _hub() -> Any:
    try:
        import huggingface_hub  # noqa: PLC0415
    except ImportError as exc:
        raise HubError("downloading from the Hugging Face Hub (library objects, texture sets) needs huggingface_hub, "
                       "which the base install does not include",
                       'pip install "alloy-engine[import]"   (or the downloader alone: pip install huggingface_hub)'
                       ) from exc
    return huggingface_hub


def explain(exc: Exception, repo: str, what: str) -> HubError:
    """A ``HubError`` with the fix for a huggingface_hub failure (private, gated, missing, offline, other)."""
    _hub()
    from huggingface_hub import errors  # noqa: PLC0415

    name = type(exc).__name__
    status = getattr(getattr(exc, "response", None), "status_code", None)
    # offline first: LocalEntryNotFoundError (not cached, no network) is a subclass of EntryNotFoundError
    if isinstance(exc, (errors.LocalEntryNotFoundError, ConnectionError, TimeoutError)):
        return HubError(f"cannot reach the Hugging Face Hub for {repo} ({what}): {name}",
                        "check the network (or HF_ENDPOINT for a mirror) and retry; files already fetched are kept")
    if isinstance(exc, errors.GatedRepoError) or status == 403:
        return HubError(f"no access to the object library {repo} ({what})", ACCESS_FIX)
    if isinstance(exc, errors.RepositoryNotFoundError) or status == 401:
        return HubError(f"the object library {repo} was not found, or it is private and this machine has no access "
                        f"({what})", ACCESS_FIX)
    if isinstance(exc, errors.EntryNotFoundError) or status == 404:
        return HubError(f"{what} is not in the object library {repo}",
                        f"update Alloy, or set {REVISION_ENV} to the dataset revision your Alloy expects")
    return HubError(f"reading {what} from {repo} failed: {name}: {exc}"[:500],
                    "retry; if it fails again, report it with this message")


def download(path: str, local_dir: Path, repo: str | None = None) -> Path:
    """Download one dataset file to ``local_dir/<path>`` and return its local path. Raises ``HubError`` with the
    fix. Example: ``download("index.jsonl", Path("/tmp/x"))``."""
    hub, repo = _hub(), repo or repo_id()
    try:
        out = hub.hf_hub_download(repo, path, repo_type=REPO_TYPE, revision=revision(), local_dir=str(local_dir))
    except Exception as exc:  # noqa: BLE001 - every Hub failure becomes a HubError with its fix
        raise explain(exc, repo, path) from exc
    return Path(out)


def load_index(cache: Path | None = None, repo: str | None = None, max_age_s: float = 0.0) -> list[dict[str, Any]]:
    """Every object row of the dataset's ``index.jsonl``. It is re-checked against the Hub on each call (unchanged
    files are not downloaded again) unless the copy in ``cache`` (default the repo's folder in Alloy's cache) was
    checked less than ``max_age_s`` ago; offline, that last copy is used. Example:
    ``[r["name"] for r in load_index() if r["library"] == "ycb_gso"]``."""
    from alloy.assets.hub.fetch import hub_dir  # noqa: PLC0415

    repo = repo or repo_id()
    cache = Path(cache) if cache is not None else hub_dir(repo)
    path, checked = cache / INDEX, cache / INDEX_CHECKED
    if max_age_s > 0 and path.is_file() and checked.is_file() and time.time() - checked.stat().st_mtime < max_age_s:
        return read_index(path)
    try:
        path = download(INDEX, cache, repo)
        checked.touch()                                    # checked now (the index file's own mtime is the Hub's)
    except HubError as exc:
        if not path.is_file() or "reach" not in exc.message:
            raise
        log.warning("%s; using the index fetched earlier", exc.message)
    return read_index(path)


def read_index(path: Path) -> list[dict[str, Any]]:
    """The object rows of an ``index.jsonl`` file. Example: ``len(read_index(hub_dir(repo_id()) / INDEX))``."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if "name" in r]
