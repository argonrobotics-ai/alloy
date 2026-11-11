# SPDX-License-Identifier: Apache-2.0
"""create from a photo or a text without a GPU here: the generation server's protocol (served by
alloy.create.serve with the generator replaced by a stand-in), the remote client, the token, and the errors that say
what to set. The whole create path from text runs when the importers' extra is installed."""

from __future__ import annotations

import base64
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from alloy.create import generate as G
from alloy.create import serve
from alloy.create.request import CreateError

BOX_OBJ = ("v -1 -1 -1\nv -1 -1 1\nv -1 1 -1\nv -1 1 1\nv 1 -1 -1\nv 1 -1 1\nv 1 1 -1\nv 1 1 1\n"
           "f 1 3 7\nf 1 7 5\nf 2 6 8\nf 2 8 4\nf 1 5 6\nf 1 6 2\n"
           "f 3 4 8\nf 3 8 7\nf 1 2 4\nf 1 4 3\nf 5 7 8\nf 5 8 6\n")
PNG_1PX = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


@pytest.fixture()
def server(monkeypatch, tmp_path):
    def fake_local(*, photo, text, work, seed=0):
        (Path(work) / "mesh.obj").write_text(BOX_OBJ)
        return {"mesh": str(Path(work) / "mesh.obj"), "image": None, "prompt": text, "seconds": 0.1}

    monkeypatch.setattr("alloy.create.gen3d_local.generate_local", fake_local)
    monkeypatch.setattr(serve._Handler, "token", "secret")          # the server's token
    monkeypatch.setenv(G.TOKEN_ENV, "secret")                         # the client's
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve._Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv(G.URL_ENV, f"http://127.0.0.1:{httpd.server_address[1]}")
    yield httpd
    httpd.shutdown()


def test_remote_text_and_photo(server, tmp_path):
    g = G.generate(text="a wooden block", work=tmp_path / "t")
    assert Path(g["mesh"]).read_text() == BOX_OBJ and g["backend"] == "remote"
    assert g["models"]["image_to_3d"]["licence"] == "MIT" and g["prompt"] == "a wooden block"
    photo = tmp_path / "p.png"
    photo.write_bytes(base64.b64decode(PNG_1PX))
    assert Path(G.generate(photo=str(photo), work=tmp_path / "p")["mesh"]).is_file()


def test_wrong_token_and_no_generator_say_what_to_do(server, monkeypatch, tmp_path):
    monkeypatch.setenv(G.TOKEN_ENV, "wrong")
    with pytest.raises(CreateError):
        G.generate(text="x", work=tmp_path)
    monkeypatch.delenv(G.URL_ENV)
    monkeypatch.setattr(G, "_local_ready", lambda: False)
    with pytest.raises(CreateError) as e:
        G.generate(text="x", work=tmp_path)
    assert G.URL_ENV in e.value.fix and "alloy.create.serve" in e.value.fix


def test_server_rejects_bad_requests():
    assert "error" in serve.handle({})
    assert "error" in serve.handle({"text": "a", "photo_b64": "AA=="})


def test_text_to_card_through_the_server(server, tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("coacd")
    pytest.importorskip("fast_simplification")
    from alloy.create.build import create

    r = create(text="a small wooden block", size=0.05, material="wood", out=str(tmp_path / "lib"), check=False)
    d = r.to_dict()
    assert d["generator"]["backend"] == "remote" and d["values"][0]["value"] == pytest.approx(0.05)
    assert "shape" in d["estimates"] and d["generator"]["leveled_deg"] == 0.0     # an upright box stays as it is
    assert d["generator"]["closed"] is True
    assert (tmp_path / "lib/objects" / f"{r.name}.yaml").is_file()
