# SPDX-License-Identifier: Apache-2.0
"""Render the camera's images for the render quality gates (``bench/render_gates``), in the gate's layout.

For each frame of a scene pack: the raw tracer at 1, 4, 16, 64 and 128 spp, denoised with OIDN ("on", as Cycles'
references) and not ("off"), plus a converged 4,096-spp image (``alloy/``); and the claimed end-to-end setting
(learned denoiser, ``alloy_e2e/``). PNGs go through the gate's own display transform with the reference folder's
``view.json``, so both sides use identical code. Optional speed rows time the claimed setting over many worlds.

Run (JSON lines on stdout; non-zero exit with the reason on stderr when a frame fails)::

    python -m alloy.camera.gate_render --scene c3 --res 128x128 --frames 420,4400 --pack packs/c3 \\
        --out gate_root --gpu l4 --rows speed.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

from .camera import Camera, load_pack
from .errors import CameraError
from .gate_claims import Claim, claim_for, parse_claim

log = logging.getLogger(__name__)
RAW_SPP = (1, 4, 16, 64, 128)
CONVERGED_SPP = 4096


def png_converter():
    """The gate's display transform ``to_png(linear (H, W, 3), view: dict) -> uint8``; it must be the function the
    references were made with (AgX looks, white balance), so it comes from bench.render_gates."""
    try:
        from bench.render_gates.view import to_png  # noqa: PLC0415  (gate tool, not shipped with the camera)
    except ImportError as exc:
        raise CameraError("bench.render_gates.view.to_png is missing",
                          "run from an Alloy checkout whose bench/render_gates has the shared view transform") from exc
    return to_png


def _converged(cam: Camera, scene, total: int, chunk: int = 256) -> tuple[np.ndarray, ...]:
    """Mean of ``total`` samples (rgb, albedo, normal) as float64 numpy (1, H, W, 3), in chunks."""
    cam.render(scene, output="linear")  # binds the renderer and uploads the camera pose
    _, r = cam._bind(scene)
    acc = [np.zeros((r.n_img, r.h, r.w, 3)) for _ in range(3)]
    done = 0
    while done < total:
        k = min(chunk, total - done)
        for a, o in zip(acc, r.render(k, done, total)):
            a += o.numpy().reshape(acc[0].shape) * k
        done += k
    return tuple(a / total for a in acc)


def _write(path: Path, linear: np.ndarray, view: dict, to_png) -> None:
    from .view import write_png
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, to_png(np.asarray(linear, np.float32), view))


def _denoise_oidn(dn, lin: np.ndarray, alb: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    arrs = [wp.array(np.ascontiguousarray(a, np.float32), dtype=wp.vec3, device=dn.device) for a in (lin, alb, nrm)]
    return dn.denoise(*arrs).numpy()


def learned(width: int, height: int, tta: int, weights: str | None = None):
    """The learned denoiser for this size at the claimed test-time augmentation (flip averaging); ``weights`` is a
    manifest name (e.g. weights fine-tuned for the scene, see :func:`alloy.camera.denoise.weights_for`)."""
    from .denoise import LearnedDenoiser
    return LearnedDenoiser(weights or LearnedDenoiser.model_for(width, height), tta=tta)


def render_frame(scene, res: str, frame_dir: Path, view: dict, to_png, opts: dict) -> list[dict]:
    """Every raw and end-to-end image of one frame; returns one row per image. ``opts``: device, backend, weights,
    claim (:class:`~alloy.camera.gate_claims.Claim`)."""
    from .oidn import OidnDenoiser
    w, h = (int(v) for v in res.split("x"))
    cam = Camera("gate", width=w, height=h, spp=1, backend=opts.get("backend"))  # one renderer for every image
    dn, rows = OidnDenoiser(w, h, device=opts["device"]), []
    for spp in RAW_SPP:
        cam.spp = spp
        out = cam.render(scene, output="all")
        lin, alb, nrm = (a.numpy().copy() for a in (out.linear, out.albedo, out.normal))
        _write(frame_dir / "alloy" / f"{res}_{spp}spp_off.png", lin[0], view, to_png)
        _write(frame_dir / "alloy" / f"{res}_{spp}spp_on.png", _denoise_oidn(dn, lin, alb, nrm)[0], view, to_png)
        rows.append({"engine": "alloy", "res": res, "spp": spp})
    cam.spp = 1
    lin, alb, nrm = _converged(cam, scene, CONVERGED_SPP)
    _write(frame_dir / "alloy" / f"{res}_{CONVERGED_SPP}spp_off.png", lin[0], view, to_png)
    _write(frame_dir / "alloy" / f"{res}_{CONVERGED_SPP}spp_on.png", _denoise_oidn(dn, lin, alb, nrm)[0], view, to_png)
    claim: Claim = opts["claim"]
    cam.spp = claim.spp
    out = cam.render(scene, output="all")
    if claim.kind == "oidn":
        e2e = _denoise_oidn(dn, *(a.numpy() for a in (out.linear, out.albedo, out.normal)))
    else:
        e2e = learned(w, h, claim.tta, _claim_weights(claim, opts))(out.linear, out.albedo, out.normal,
                                                                     claim.spp).numpy()
    dn.close()
    _write(frame_dir / "alloy_e2e" / f"{res}_{claim.spp}spp_on.png", e2e[0], view, to_png)
    rows.append({"engine": "alloy_e2e", "res": res, "spp": claim.spp, "denoiser": claim.kind,
                 "weights": (_claim_weights(claim, opts) or "default") if claim.kind == "learned" else None})
    return rows


def speed_row(pack: Path, frame: int, res: str, worlds: int, gpu: str, opts: dict) -> dict:
    """images/s at the claimed setting over ``worlds`` worlds (render + denoiser + display), GPU alone. When that many
    worlds do not fit in GPU memory, the world count halves until it does: the row then has the ``worlds`` measured
    and ``worlds_requested`` (a speed bar at the requested count stays unmet).
    ``opts``: device, backend, weights, claim.
    """
    n = int(worlds)
    while True:
        try:
            row = _speed_row_at(pack, frame, res, n, gpu, opts)
            return row if n == worlds else {**row, "worlds_requested": int(worlds), "fits_requested": False}
        except RuntimeError as exc:  # Warp "Failed to allocate", torch.OutOfMemoryError
            if not _out_of_memory(exc) or n == 1:
                raise
            log.warning("speed row %s at %d worlds ran out of GPU memory; trying %d", res, n, n // 2)
            n //= 2
            _release_gpu_memory()


def _out_of_memory(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "failed to allocate" in text or "out of memory" in text


def _release_gpu_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _speed_row_at(pack: Path, frame: int, res: str, worlds: int, gpu: str, opts: dict) -> dict:
    from .denoise import DenoiseSpec
    from .wavefront.tracers import resolve_backend
    w, h = (int(v) for v in res.split("x"))
    claim: Claim = opts["claim"]
    device = opts["device"]
    scene = load_pack(pack, frame=frame, num_worlds=worlds, device=device, cache=False)
    weights = _claim_weights(claim, opts) if claim.kind == "learned" else None
    cam = Camera("speed", width=w, height=h, spp=claim.spp, backend=opts.get("backend"),
                 denoise=DenoiseSpec(claim.kind, weights=weights, tta=claim.tta))
    for _ in range(2):
        cam.render(scene)
    wp.synchronize_device(device)
    t0, n = time.perf_counter(), 5
    for _ in range(n):
        cam.render(scene)
    wp.synchronize_device(device)
    dt = (time.perf_counter() - t0) / n
    return {"gpu": gpu, "res": res, "worlds": worlds, "shared_gpu": False, "setting": "claim",
            "images_per_s": round(worlds / dt, 1), "spp": claim.spp, "denoiser": claim.kind, "tta": claim.tta,
            "weights": (weights or "default") if claim.kind == "learned" else None,
            "backend": resolve_backend(cam._bound[1].st.backend, device)}


def _claim_weights(claim: Claim, opts: dict) -> str | None:
    """Learned weights for the claim: ``--weights`` when given explicitly, else the claim's own, else the scene's."""
    return opts.get("weights_override") or claim.weights or opts.get("weights")


def _weights(a) -> str | None:
    """The --weights choice as a manifest name, or None for the size's default model."""
    if a.weights == "default":
        return None
    if a.weights != "auto":
        return a.weights
    from .denoise import LearnedDenoiser, weights_for
    w, h = (int(v) for v in a.res.split("x"))
    name = weights_for(a.scene, w, h)
    return None if name == LearnedDenoiser.model_for(w, h) else name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--frames", required=True, help="comma-separated frame numbers")
    ap.add_argument("--out", required=True, type=Path, help="gate root")
    ap.add_argument("--pack", required=True, type=Path, help="the scene's pack folder (v2 pack.json or v1 scene.json)")
    ap.add_argument("--gpu", default="")
    ap.add_argument("--rows", type=Path, default=None, help="append a speed row (JSONL) at the claimed setting")
    ap.add_argument("--worlds", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--backend", default="auto", help="warp | optix | auto (optix when built, else warp)")
    ap.add_argument("--claim", default=None, help="score a candidate instead of the scene's claim: "
                    "SPP,KIND[,TTA[,WEIGHTS]], e.g. 16,learned,1,c3_1280 or 64,oidn")
    ap.add_argument("--weights", default="auto",
                    help="learned-denoiser weights: auto (the scene's own when the manifest has them), default, "
                         "or a name")
    a = ap.parse_args(argv)
    try:
        to_png, pack = png_converter(), a.pack
        opts = {"device": a.device, "backend": a.backend, "weights": _weights(a),
                "claim": parse_claim(a.claim) if a.claim else claim_for(a.scene, a.res),
                "weights_override": a.weights if a.weights not in ("auto", "default") else None}
        log.info("claim for %s at %s: %s", a.scene, a.res, opts["claim"].label())
        for f in a.frames.split(","):
            fdir = a.out / a.scene / f
            vj = fdir / "cycles" / "view.json"
            view = json.loads(vj.read_text()) if vj.exists() else {}  # T12: Standard
            scene = load_pack(pack, frame=int(f), device=a.device)
            for row in render_frame(scene, a.res, fdir, view, to_png, opts):
                print(json.dumps({"scene": a.scene, "frame": f, **row}), flush=True)
            gc.collect()  # free the frame's GPU buffers now (Warp's kernel decorators leave collectable cycles)
    except (CameraError, OSError, KeyError, ValueError) as exc:
        print(f"gate_render {a.scene} {a.res}: {exc}", file=sys.stderr)
        return 1
    if a.rows:  # a speed row that cannot be measured does not fail the quality images already written
        try:
            row = {"scene": a.scene, **speed_row(pack, int(a.frames.split(",")[0]), a.res, a.worlds, a.gpu, opts)}
        except (CameraError, RuntimeError, OSError) as exc:
            print(f"gate_render {a.scene} {a.res}: no speed row: {exc}", file=sys.stderr)
            return 0
        with a.rows.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
