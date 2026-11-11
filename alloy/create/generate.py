# SPDX-License-Identifier: Apache-2.0
"""A mesh from a photo or a text description, for ``alloy.create``.

Models (all allow commercial use; checked 2026-11-06):

| Step | Model | Licence |
|---|---|---|
| text -> image | Kandinsky 2.2 (kandinsky-community/kandinsky-2-2-decoder + prior) | Apache-2.0 |
| text -> image, optional | FLUX.1 [schnell] (black-forest-labs/FLUX.1-schnell; gated: needs a HF login) | Apache-2.0 |
| background removal | rembg with U^2-Net | MIT (code), Apache-2.0 (weights) |
| image -> 3D | TripoSR (stabilityai/TripoSR) | MIT (code and weights) |
| marching cubes | scikit-image (in place of TripoSR's torchmcubes, MPL-2.0) | BSD-3-Clause |

They need an NVIDIA GPU (about 6 GB for TripoSR, 10 GB for Kandinsky; FLUX with CPU offload fits a 24 GB GPU and
64 GB RAM). Two ways:

- **remote**: set ``$ALLOY_GEN3D_URL`` to a generation server (``python -m alloy.create.serve`` on a GPU machine);
  this machine needs no GPU. ``$ALLOY_GEN3D_TOKEN`` is sent as a bearer token if set;
- **local**: on a GPU machine with the generator installed (``alloy.create.gen3d_local``), it runs in process.

The generated mesh has no real size; ``alloy.create`` scales it (your ``size=``, else the category's typical size).

Example::

    >>> g = generate(text="a red ceramic coffee mug", work=Path("tmp/gen"))      # doctest: +SKIP
    >>> g["mesh"], g["models"]["image_to_3d"]["licence"]                         # doctest: +SKIP
    ('tmp/gen/mesh.obj', 'MIT')
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

from alloy.create.remote import TOKEN_ENV, URL_ENV, post_generate
from alloy.create.request import CreateError

log = logging.getLogger("alloy.create")

MODELS: dict[str, dict[str, str]] = {
    "text_to_image": {"name": "Kandinsky 2.2", "repo": "kandinsky-community/kandinsky-2-2-decoder",
                      "licence": "Apache-2.0"},
    "background": {"name": "rembg U^2-Net", "repo": "danielgatis/rembg", "licence": "MIT; weights Apache-2.0"},
    "image_to_3d": {"name": "TripoSR", "repo": "stabilityai/TripoSR", "licence": "MIT"},
    "marching_cubes": {"name": "scikit-image", "repo": "scikit-image/scikit-image", "licence": "BSD-3-Clause"},
}


def generate(*, photo: str | None = None, text: str | None = None, work: Path, seed: int = 0) -> dict[str, Any]:
    """Mesh file (OBJ, arbitrary units, +z up) for ``photo`` or ``text``: ``{"mesh", "image", "backend", "models",
    "prompt", "seconds"}``. Raises ``CreateError`` with the fix when no generator is reachable."""
    if (photo is None) == (text is None):
        raise CreateError("generate needs a photo or a text, not both", "pass photo= or text=", field="input")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    url = os.environ.get(URL_ENV, "").strip()
    if url:
        return _remote(url, photo, text, work, seed)
    if _local_ready():
        from alloy.create.gen3d_local import generate_local  # noqa: PLC0415

        return generate_local(photo=photo, text=text, work=work, seed=seed)
    raise CreateError("no 3D generator here: no $ALLOY_GEN3D_URL and no NVIDIA GPU with the generator installed",
                      f"run `python -m alloy.create.serve` on a GPU machine and set {URL_ENV}=http://<host>:8765, or "
                      "pass a mesh file (mesh=) instead", field="input")


def _local_ready() -> bool:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return False
        from alloy.create.gen3d_local import available  # noqa: PLC0415

        return available()
    except Exception:  # noqa: BLE001 - any missing piece means "not here"; the error message says what to do
        return False


def _remote(url: str, photo: str | None, text: str | None, work: Path, seed: int) -> dict[str, Any]:
    body: dict[str, Any] = {"seed": seed}
    if photo is not None:
        p = Path(photo).expanduser()
        if not p.is_file():
            raise CreateError(f"photo {photo!r} not found", "pass the path of an existing image file", field="photo")
        body["photo_b64"] = base64.b64encode(p.read_bytes()).decode("ascii")
    else:
        body["text"] = text
    data = post_generate(url, body)
    if "error" in data:
        raise CreateError(f"the generation server failed: {data['error']}", data.get("fix", "check the server log"),
                          field="input")
    (work / "mesh.obj").write_bytes(base64.b64decode(data["mesh_obj_b64"]))
    if data.get("image_png_b64"):
        (work / "image.png").write_bytes(base64.b64decode(data["image_png_b64"]))
    return {"mesh": str(work / "mesh.obj"), "image": str(work / "image.png") if data.get("image_png_b64") else None,
            "backend": "remote", "models": data.get("models", MODELS), "prompt": data.get("prompt"),
            "seconds": data.get("seconds")}


__all__ = ["MODELS", "TOKEN_ENV", "URL_ENV", "generate"]
