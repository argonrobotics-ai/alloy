# SPDX-License-Identifier: Apache-2.0
"""Denoisers for camera images: the learned denoiser (``alloy.camera.denoiser``, trained on Cycles renders) with
Intel Open Image Denoise as the fallback. Both take the linear colour, the denoising albedo and the camera-space
normal of a render (Cycles' denoising-pass conventions) and return linear colour.

``kind``: "auto" (learned if its package and weights load, else OIDN if installed, else none; a fallback is said
once on the console, with the reason), "learned", "oidn", or None / "off". The learned weights download once from the
Alloy library on the Hugging Face Hub into ``~/.cache/alloy/denoiser`` (``alloy.camera.denoiser.weights``). A
:class:`DenoiseSpec` also picks the learned weights (e.g. ones fine-tuned for one scene) and the test-time
augmentation.

Example::

    from alloy.camera import Camera, DenoiseSpec
    cam = Camera("front", width=128, height=128, spp=4, denoise="auto")
    img = cam.render(model, state)                # denoised, then the display transform
    cam = Camera("front", width=128, height=128, spp=5, denoise=DenoiseSpec("learned", weights="t12", tta=2))
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import warp as wp

from .errors import CameraError

log = logging.getLogger(__name__)
KINDS = ("auto", "learned", "oidn", "off", None)


@dataclass(frozen=True)
class DenoiseSpec:
    """Which denoiser: ``kind`` (see :data:`KINDS`), the learned ``weights`` (a manifest name of
    ``alloy.camera.denoiser``, a .pt path, or None for the default model of the image size) and ``tta`` (1, 2 or 4
    flipped passes averaged by the learned denoiser; 2 helps at 4-5 spp).

    Example: ``DenoiseSpec("learned", weights="t12", tta=2)``.
    """

    kind: str | None = "auto"
    weights: str | None = None
    tta: int = 1

    def __post_init__(self):
        if self.kind not in KINDS:
            raise CameraError(f"denoise kind {self.kind!r}", f"use one of {KINDS}")
        if int(self.tta) < 1:
            raise CameraError(f"denoise tta={self.tta}", "use tta >= 1 (1 = no test-time augmentation)")


def as_spec(denoise) -> DenoiseSpec:
    """A :class:`DenoiseSpec` from a kind string, None or a spec (unchanged)."""
    return denoise if isinstance(denoise, DenoiseSpec) else DenoiseSpec(denoise)


def weights_for(scene: str, width: int, height: int) -> str:
    """The learned weights for ``scene`` at this size: a manifest entry named ``<scene>`` or
    ``<scene>_<128|640|1280>`` trained at the matching size class (``train_res`` "128" up to 256 px, "640x480" up to
    640 px, else "1280x720"), else the default model.

    Example: ``weights_for("c3", 1280, 720)`` -> "c3_1280" when the manifest has it, else "default:640x480".
    """
    default = LearnedDenoiser.model_for(width, height)
    try:
        from alloy.camera.denoiser.weights import list_weights  # noqa: PLC0415  (optional package)
        known = list_weights()
    except (ImportError, OSError, ValueError, KeyError) as exc:
        log.info("denoiser manifest unavailable (%s); using %s", exc, default)
        return default
    side = max(width, height)
    res = "128" if side <= 256 else "640x480" if side <= 640 else "1280x720"
    for name in (scene, f"{scene}_{res.split('x')[0]}"):
        if known.get(name, {}).get("train_res") == res:
            return name
    return default


class LearnedDenoiser:
    """Adapter for OSS-6's ``alloy.camera.denoiser.load(name).denoise(rgb, albedo, normal, spp)`` (torch tensors)."""

    def __init__(self, name: str = "default", tta: int = 1):
        from alloy.camera.denoiser import load  # noqa: PLC0415  (optional package, needs PyTorch)
        self.model = load(name)
        self.name, self.tta = name, int(tta)

    @staticmethod
    def model_for(width: int, height: int) -> str:
        """The weights trained for this image size: the 128 px model up to 256 px, else the 640x480 one."""
        return "default" if max(width, height) <= 256 else "default:640x480"

    def __call__(self, rgb: wp.array, albedo: wp.array, normal: wp.array, spp: int) -> wp.array:
        n, h, w = rgb.shape
        per = max(1, chunk_pixels(rgb.device) // (h * w))
        t = [wp.to_torch(a) for a in (rgb, albedo, normal)]
        parts = [self.model.denoise(*(x[i:i + per] for x in t), spp, tta=self.tta) for i in range(0, n, per)]
        out = parts[0] if len(parts) == 1 else _cat(parts)
        return wp.from_torch(out.contiguous().float(), dtype=wp.vec3)


def chunk_pixels(device) -> int:
    """Pixels the learned denoiser takes in one call: about one per 4 KB of GPU memory (5.5 M on a 24 GB L4, all
    1,024 images of 128 px at once on a 96 GB GPU), so big batches run in chunks instead of running out of memory."""
    return max(1 << 18, int(wp.get_device(device).total_memory) // 4096)


def _cat(parts):
    import torch  # noqa: PLC0415  (the learned denoiser needs PyTorch anyway)
    return torch.cat(parts, dim=0)


class OidnAdapter:
    """OIDN with Cycles' settings (see ``alloy.camera.oidn``); ``spp`` is not needed."""

    def __init__(self, width: int, height: int, device: str):
        from .oidn import OidnDenoiser
        self.dn = OidnDenoiser(width, height, device=device)
        self._out = None

    def __call__(self, rgb: wp.array, albedo: wp.array, normal: wp.array, spp: int) -> wp.array:
        if self._out is None or self._out.shape != rgb.shape:
            self._out = wp.empty(rgb.shape, dtype=wp.vec3, device=rgb.device)
        return self.dn.denoise(rgb, albedo, normal, self._out)


_SAID: set[str] = set()


def _say_once(msg: str) -> None:
    """Tell the user ``msg`` once per process (shown on the console without any logging setup)."""
    if msg in _SAID:
        return
    _SAID.add(msg)
    from alloy.api._say import warn  # noqa: PLC0415 - the API's user warning (the package logger is silent)
    warn("%s", msg)


def _why(exc: Exception) -> str:
    """One line on why the learned denoiser did not load."""
    if isinstance(exc, ImportError):
        return f"PyTorch is missing ({exc}): pip install torch"
    return str(exc).split(". ")[0][:400]


def make_denoiser(denoise, width: int, height: int, device: str):
    """A callable ``(rgb, albedo, normal, spp) -> rgb`` for ``denoise`` (a kind or a :class:`DenoiseSpec`), or None
    for no denoising. With "auto", a fallback (OIDN, or no denoising) is said once per process, with the reason."""
    spec = as_spec(denoise)
    kind = spec.kind
    if kind in (None, "off"):
        return None
    if not wp.get_device(device).is_cuda:
        if kind != "auto":
            raise CameraError(f"denoise={kind!r} on {device}", "denoise on a CUDA device, or use denoise=None")
        _say_once(f"denoise='auto': the denoisers need a CUDA device, so images on {device} are not denoised")
        return None
    learned = ""
    if kind in ("auto", "learned"):
        try:
            return LearnedDenoiser(spec.weights or LearnedDenoiser.model_for(width, height), tta=spec.tta)
        except Exception as exc:  # noqa: BLE001  (optional package / weights / torch)
            if kind == "learned":
                raise CameraError(f"the learned denoiser did not load: {exc}",
                                  "install PyTorch, let the weights download once (or see "
                                  "alloy.camera.denoiser.weights for offline use), or use denoise='oidn'") from exc
            learned = f"the learned denoiser did not load ({_why(exc)})"
    from .oidn import available
    ok, why = available()
    if ok:
        if learned:
            _say_once(f"denoise='auto': {learned}; using Open Image Denoise instead")
        return OidnAdapter(width, height, device)
    if kind == "oidn":
        raise CameraError("OIDN is not installed", why.split(": ", 1)[-1])
    _say_once(f"denoise='auto': {learned or 'no learned denoiser'}, and Open Image Denoise is not installed "
              f"({why.split(': ', 1)[-1]}); images are not denoised")
    return None
