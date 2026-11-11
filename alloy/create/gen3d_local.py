# SPDX-License-Identifier: Apache-2.0
"""The 3D generator running in this process (NVIDIA GPU): text -> image with Kandinsky 2.2 (or FLUX.1 [schnell]
with ``$ALLOY_GEN3D_T2I=flux`` and a Hugging Face login that has accepted its terms), background removal
with rembg, image -> mesh with TripoSR, marching cubes with scikit-image. See ``alloy.create.generate`` for licences.

Install on the GPU machine (Python with CUDA PyTorch): ``pip install diffusers transformers accelerate
sentencepiece rembg onnxruntime scikit-image einops omegaconf trimesh huggingface_hub`` and a TripoSR checkout
(``git clone https://github.com/VAST-AI-Research/TripoSR`` at ``TRIPOSR_COMMIT``) on ``$ALLOY_TRIPOSR_DIR``. TripoSR's
``torchmcubes`` (MPL-2.0) is not needed: ``_mc_shim`` stands in for it with scikit-image.

Example::

    >>> generate_local(text="a wooden spoon", work=Path("gen"))      # doctest: +SKIP
"""

from __future__ import annotations

import logging
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("alloy.create")

TRIPOSR_ENV = "ALLOY_TRIPOSR_DIR"
TRIPOSR_COMMIT = "107cefdc244c39106fa830359024f6a2f1c78871"    # VAST-AI-Research/TripoSR main, 2026-11-06
KANDINSKY = "kandinsky-community/kandinsky-2-2-decoder"      # Apache-2.0, not gated (the default)
FLUX = "black-forest-labs/FLUX.1-schnell"                   # Apache-2.0 but gated: needs a Hugging Face login
T2I_ENV = "ALLOY_GEN3D_T2I"                                  # "kandinsky" (default) or "flux"
TRIPOSR = "stabilityai/TripoSR"
MC_RESOLUTION = 256
FOREGROUND = 0.85
PROMPT = ("a product photo of {text}, one single object, the whole object in view, centered, seen from the front "
          "and slightly above, plain light grey background, bright even studio light, no text, no shadow")
NEGATIVE = "dark, black background, text, watermark, cropped, several objects, hands, people, blurry, low quality"
_CACHE: dict[str, Any] = {}


def available() -> bool:
    """True when TripoSR, diffusers, rembg and scikit-image import here."""
    try:
        _tsr_path()
        import diffusers  # noqa: F401, PLC0415
        import rembg  # noqa: F401, PLC0415
        import skimage  # noqa: F401, PLC0415
        import tsr  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001 - not installed here
        return False
    return True


def generate_local(*, photo: str | None, text: str | None, work: Path, seed: int = 0) -> dict[str, Any]:
    """See ``alloy.create.generate.generate`` (same return)."""
    from PIL import Image  # noqa: PLC0415

    from alloy.create.generate import MODELS  # noqa: PLC0415

    t0 = time.perf_counter()
    prompt = None
    if photo is not None:
        image = Image.open(photo).convert("RGBA")
    else:
        prompt = PROMPT.format(text=text.strip().rstrip("."))
        image = _text_to_image(prompt, seed)
    image.save(work / "image.png")
    mesh = _image_to_mesh(image)
    mesh.export(work / "mesh.obj")
    return {"mesh": str(work / "mesh.obj"), "image": str(work / "image.png"), "backend": "local", "models": MODELS,
            "prompt": prompt, "seconds": round(time.perf_counter() - t0, 1)}


def loaded_models() -> list[str]:
    """The models already in memory in this process (``kandinsky``, ``flux``, ``rembg``, ``tsr``)."""
    return sorted(_CACHE)


def text_to_image_model() -> str:
    """``kandinsky`` (default) or ``flux`` (``$ALLOY_GEN3D_T2I``)."""
    return "flux" if os.environ.get(T2I_ENV, "").strip().lower() == "flux" else "kandinsky"


def _text_to_image(prompt: str, seed: int):
    import torch  # noqa: PLC0415

    which = text_to_image_model()
    if which not in _CACHE:
        _CACHE[which] = _flux() if which == "flux" else _kandinsky()
    gen = torch.Generator("cpu").manual_seed(seed)
    if which == "flux":
        return _CACHE[which](prompt, guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256,
                             height=768, width=768, generator=gen).images[0]
    return _CACHE[which](prompt=prompt, negative_prompt=NEGATIVE, num_inference_steps=40, prior_num_inference_steps=25,
                         height=768, width=768, generator=gen).images[0]


def _kandinsky():
    import torch  # noqa: PLC0415
    from diffusers import AutoPipelineForText2Image  # noqa: PLC0415

    return AutoPipelineForText2Image.from_pretrained(KANDINSKY, torch_dtype=torch.float16).to("cuda")


def _flux():
    import torch  # noqa: PLC0415
    from diffusers import FluxPipeline  # noqa: PLC0415

    pipe = FluxPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe


def _image_to_mesh(image):
    """TripoSR on the cut-out object; returns a trimesh with +z up and outward faces."""
    import torch  # noqa: PLC0415

    model = _triposr()
    rgb = _cutout(image)
    with torch.no_grad():
        codes = model([rgb], device="cuda")
    mesh = model.extract_mesh(codes, True, resolution=MC_RESOLUTION)[0]
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _cutout(image):
    """Background removed, object resized to fill ``FOREGROUND`` of the frame, on mid grey (TripoSR's input)."""
    import rembg  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from tsr.utils import resize_foreground  # noqa: PLC0415

    if "rembg" not in _CACHE:
        _CACHE["rembg"] = rembg.new_session()
    cut = resize_foreground(rembg.remove(image, session=_CACHE["rembg"]), FOREGROUND)
    a = np.asarray(cut).astype(np.float32) / 255.0
    rgb = a[:, :, :3] * a[:, :, 3:4] + (1.0 - a[:, :, 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).astype(np.uint8))


def _triposr():
    if "tsr" not in _CACHE:
        _tsr_path()
        _mc_shim()
        _CACHE["tsr"] = _load_triposr()
    return _CACHE["tsr"]


def _load_triposr():
    """``TSR.from_pretrained`` with the checkpoint's ViT keys renamed for the installed ``transformers``."""
    import torch  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415
    from tsr.system import TSR  # noqa: PLC0415

    cfg = OmegaConf.load(hf_hub_download(TRIPOSR, "config.yaml"))
    OmegaConf.resolve(cfg)
    model = TSR(cfg)
    ckpt = torch.load(hf_hub_download(TRIPOSR, "model.ckpt"), map_location="cpu")
    model.load_state_dict(remap_vit(ckpt, set(model.state_dict())))
    model.renderer.set_chunk_size(8192)
    return model.to("cuda")


#: transformers < 5 ViT names -> transformers 5 names (TripoSR's DINO image tokenizer)
_VIT_RENAMES = ((".encoder.layer.", ".layers."), (".attention.attention.query.", ".attention.q_proj."),
                (".attention.attention.key.", ".attention.k_proj."),
                (".attention.attention.value.", ".attention.v_proj."),
                (".attention.output.dense.", ".attention.o_proj."), (".intermediate.dense.", ".mlp.fc1."),
                (".output.dense.", ".mlp.fc2."))


def remap_vit(ckpt: dict, model_keys: set[str]) -> dict:
    """Rename checkpoint keys the model does not have to the names it does (a no-op when they already match)."""
    out = {}
    for k, v in ckpt.items():
        new = k
        if k not in model_keys:
            for a, b in _VIT_RENAMES:
                new = new.replace(a, b)
        out[new if new in model_keys else k] = v
    return out


def _tsr_path() -> None:
    d = os.environ.get(TRIPOSR_ENV, "").strip()
    if d and d not in sys.path:
        sys.path.insert(0, d)


def _mc_shim() -> None:
    """A ``torchmcubes`` module backed by scikit-image (BSD), with torchmcubes' (x = last axis) vertex order."""
    if "torchmcubes" in sys.modules:
        return
    import torch  # noqa: PLC0415

    def marching_cubes(vol, thresh):
        from skimage.measure import marching_cubes as mc  # noqa: PLC0415

        v, f, _, _ = mc(vol.detach().float().cpu().numpy(), level=float(thresh))
        v = np.ascontiguousarray(v[:, ::-1], dtype=np.float32)
        return torch.from_numpy(v), torch.from_numpy(np.ascontiguousarray(f, dtype=np.int64))

    mod = types.ModuleType("torchmcubes")
    mod.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = mod


__all__ = ["TRIPOSR_COMMIT", "available", "generate_local", "loaded_models"]
