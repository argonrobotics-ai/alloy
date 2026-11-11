# SPDX-License-Identifier: Apache-2.0
"""Denoiser selection and the OIDN fallback (the learned denoiser is tested by its own package)."""
from __future__ import annotations

import numpy as np
import pytest

from alloy.camera import Camera, CameraError, load_pack
from alloy.camera.denoise import DenoiseSpec, as_spec, make_denoiser, weights_for
from alloy.camera.oidn import available
from tests.camera.packs import principled_material, write_pack_v2


def test_selection_rules():
    assert make_denoiser(None, 8, 8, "cpu") is None and make_denoiser("off", 8, 8, "cpu") is None
    assert make_denoiser("auto", 8, 8, "cpu") is None  # no CUDA: render without denoising
    with pytest.raises(CameraError, match="use one of"):
        make_denoiser("magic", 8, 8, "cpu")
    with pytest.raises(CameraError, match="CUDA device"):
        make_denoiser("oidn", 8, 8, "cpu")
    with pytest.raises(CameraError, match="use one of"):
        Camera("c", denoise="blur")


def test_denoise_spec_picks_weights_and_tta():
    assert as_spec("oidn") == DenoiseSpec("oidn") and as_spec(None).kind is None
    spec = DenoiseSpec("learned", weights="t12", tta=2)
    assert as_spec(spec) is spec and Camera("c", denoise=spec).denoise is spec
    with pytest.raises(CameraError, match="tta >= 1"):
        DenoiseSpec("learned", tta=0)
    with pytest.raises(CameraError, match="use one of"):
        DenoiseSpec("blur")


def test_weights_for_prefers_the_scenes_own(monkeypatch):
    import alloy.camera.denoiser.weights as W
    known = {"t12": {"train_res": "128"}, "c3_640": {"train_res": "640x480"}, "v0_128": {"train_res": "128"},
             "c3_1280": {"train_res": "1280x720"}}
    monkeypatch.setattr(W, "list_weights", lambda manifest=None: known)
    assert weights_for("t12", 128, 128) == "t12"
    assert weights_for("t12", 640, 480) == "default:640x480"   # t12 weights are 128 px only
    assert weights_for("c3", 640, 480) == "c3_640"
    assert weights_for("c3", 128, 128) == "default"
    assert weights_for("c3", 1280, 720) == "c3_1280" and weights_for("t12", 1280, 720) == "default:640x480"


def test_learned_denoiser_runs_big_batches_in_chunks(monkeypatch):
    torch = pytest.importorskip("torch")
    import warp as wp

    import alloy.camera.denoise as D
    seen = []

    class Halve:
        def denoise(self, rgb, albedo, normal, spp, tta=1):
            seen.append(rgb.shape[0])
            return rgb * 0.5

    dn = D.LearnedDenoiser.__new__(D.LearnedDenoiser)
    dn.model, dn.name, dn.tta = Halve(), "stub", 1
    monkeypatch.setattr(D, "chunk_pixels", lambda device: 3 * 4 * 4)  # three 4x4 images per call
    rgb = wp.array(np.arange(7 * 4 * 4 * 3, dtype=np.float32).reshape(7, 4, 4, 3), dtype=wp.vec3, device="cpu")
    out = dn(rgb, rgb, rgb, 4)
    assert seen == [3, 3, 1] and torch.allclose(wp.to_torch(out), wp.to_torch(rgb) * 0.5)


def test_oidn_availability_says_how_to_install(monkeypatch, tmp_path):
    monkeypatch.setenv("OIDN_LIB_DIR", str(tmp_path))
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path))
    ok, why = available()  # may still find a system-wide OIDN
    if not ok:
        assert "OIDN_LIB_DIR" in why and "python -m alloy.camera.oidn fetch" in why


@pytest.mark.gpu
def test_oidn_denoises_the_furnace_plane(tmp_path):
    if not available()[0]:
        pytest.skip("OIDN not installed")
    scene = load_pack(write_pack_v2(tmp_path / "f", principled_material((0.5, 0.5, 0.5), specular=0.5), sky=0.5),
                      device="cuda:0")
    raw = Camera("raw", width=32, height=32, spp=1).render(scene, output="linear").numpy().copy()
    clean = Camera("dn", width=32, height=32, spp=1, denoise="oidn").render(scene, output="all")
    lin = clean.linear.numpy()
    assert np.isfinite(lin).all() and lin.std() < 0.5 * raw.std()
    assert abs(lin.mean() - raw.mean()) < 0.05 * raw.mean()
    assert np.array_equal(clean.noisy.numpy(), raw)  # same seed: the raw colour is kept alongside


class _Cuda:
    is_cuda = True


def test_auto_says_once_what_it_fell_back_to(monkeypatch, caplog, capsys):
    """denoise="auto" names its fallback and the reason, once per process (audit AB-camera-3): a user warning,
    logged to ``alloy.api`` and printed to stderr when no log handler shows it (here pytest's handler does)."""
    from alloy.camera import denoise as D
    from alloy.camera import oidn

    class NoWeights:
        model_for = staticmethod(D.LearnedDenoiser.model_for)

        def __init__(self, name="default", tta=1):
            raise FileNotFoundError("Weights 'v3' are not cached in /c and downloads are off. Copy dn_v3.pt into /c.")

    def said() -> str:
        text = caplog.text + capsys.readouterr().err
        caplog.clear()
        return text

    monkeypatch.setattr(D, "_SAID", set())
    assert make_denoiser("auto", 8, 8, "cpu") is None and make_denoiser("auto", 8, 8, "cpu") is None
    text = said()
    assert text.count("denoise='auto'") == 1 and "need a CUDA device" in text and "not denoised" in text

    monkeypatch.setattr(D.wp, "get_device", lambda device: _Cuda())
    monkeypatch.setattr(D, "LearnedDenoiser", NoWeights)
    monkeypatch.setattr(D, "OidnAdapter", lambda w, h, device: "oidn")
    monkeypatch.setattr(oidn, "available", lambda: (True, "/lib"))
    assert make_denoiser("auto", 8, 8, "cuda:0") == "oidn" and make_denoiser("auto", 8, 8, "cuda:0") == "oidn"
    text = said()
    assert text.count("denoise='auto'") == 1 and "using Open Image Denoise instead" in text
    assert "Weights 'v3' are not cached in /c and downloads are off" in text and "Copy dn_v3" not in text
    assert make_denoiser("oidn", 8, 8, "cuda:0") == "oidn" and said() == ""  # asked for by name: no notice

    monkeypatch.setattr(oidn, "available", lambda: (False, "Open Image Denoise not found: run the fetch command"))
    assert make_denoiser("auto", 8, 8, "cuda:0") is None and make_denoiser("auto", 8, 8, "cuda:0") is None
    text = said()
    assert text.count("denoise='auto'") == 1 and "images are not denoised" in text and "run the fetch command" in text
    with pytest.raises(CameraError, match="the learned denoiser did not load: Weights 'v3' are not cached"):
        make_denoiser("learned", 8, 8, "cuda:0")


def test_weights_come_from_the_public_library_by_default(monkeypatch, tmp_path):
    """No setting: the Hub library's denoiser/ folder over https; ALLOY_DENOISER_URL overrides, empty turns it off."""
    import hashlib

    from alloy.assets.hub.source import DEFAULT_REPO
    from alloy.camera.denoiser import weights as W

    for var in (W.URL_ENV, "HF_HUB_OFFLINE", "HF_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    assert W.HUB_REPO == DEFAULT_REPO
    assert W.download_url() == f"https://huggingface.co/datasets/{DEFAULT_REPO}/resolve/main/denoiser"
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example/")
    assert W.download_url() == f"https://mirror.example/datasets/{DEFAULT_REPO}/resolve/main/denoiser"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert W.download_url() is None and W.download_url("/data/w") == "/data/w"
    monkeypatch.delenv("HF_HUB_OFFLINE")

    src, cache, blob = tmp_path / "src", tmp_path / "cache", b"weights"
    src.mkdir()
    (src / "dn_t.pt").write_bytes(blob)
    manifest = {"default": {"128": "t"}, "weights": {"t": {"file": "dn_t.pt", "bytes": len(blob),
                                                           "sha256": hashlib.sha256(blob).hexdigest()}}}
    monkeypatch.setenv(W.URL_ENV, "")
    with pytest.raises(FileNotFoundError, match="downloads are off.*Copy dn_t.pt into"):
        W.resolve_weights("default", cache_dir=cache, manifest=manifest)
    monkeypatch.setenv(W.URL_ENV, str(src))
    got = W.resolve_weights("default", cache_dir=cache, manifest=manifest)
    assert got == cache / "dn_t.pt" and got.read_bytes() == blob
    monkeypatch.setenv(W.URL_ENV, "")  # cached: no source needed
    assert W.resolve_weights("t", cache_dir=cache, manifest=manifest) == got
    for name, entry in W.list_weights().items():  # every shipped entry is a real file of the public folder
        assert entry["sha256"] != "0" * 64 and entry["file"].startswith("dn_") and entry["file"].endswith(".pt"), name


def test_unpublished_or_unreachable_weights_are_said_plainly(monkeypatch, tmp_path, caplog, capsys):
    """Before the weights are on the Hub the address answers 404: "learned" raises one plain sentence, and "auto"
    falls back with it in its one-line notice; an unreachable host reads the same way. No traceback is printed."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from alloy.camera import denoise as D
    from alloy.camera import oidn
    from alloy.camera.denoiser import weights as W

    class NotFound(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802  (http.server's name)
            self.send_error(404)

        def log_message(self, *args):
            pass

    class Fetching:
        model_for = staticmethod(D.LearnedDenoiser.model_for)

        def __init__(self, name="default", tta=1):
            W.resolve_weights(name, cache_dir=tmp_path / "cache")

    server = ThreadingHTTPServer(("127.0.0.1", 0), NotFound)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    for var in (W.URL_ENV, "HF_HUB_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(D, "_SAID", set())
    monkeypatch.setattr(D.wp, "get_device", lambda device: _Cuda())
    monkeypatch.setattr(D, "LearnedDenoiser", Fetching)
    monkeypatch.setattr(D, "OidnAdapter", lambda w, h, device: "oidn")
    monkeypatch.setattr(oidn, "available", lambda: (True, "/lib"))
    try:
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{port}")
        with pytest.raises(CameraError, match="dn_v3.pt is not published there yet, or the address is wrong"):
            make_denoiser("learned", 8, 8, "cuda:0")
        assert make_denoiser("auto", 8, 8, "cuda:0") == "oidn"
        text = caplog.text + capsys.readouterr().err
        assert "did not load (The denoiser weights file dn_v3.pt is not published there yet" in text
        assert "answered HTTP 404" in text and "using Open Image Denoise instead" in text and "Traceback" not in text
    finally:
        server.shutdown()
        server.server_close()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{port}")  # the server is gone: connection refused
    with pytest.raises(CameraError, match="Could not reach http://127.0.0.1.*for the denoiser weights"):
        make_denoiser("learned", 8, 8, "cuda:0")
    assert not list((tmp_path / "cache").glob("*"))  # no partial file is left behind
