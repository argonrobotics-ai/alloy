# SPDX-License-Identifier: Apache-2.0
"""An elastic rod (rod_vbd) beside each other kind of object (audit COUPLE-2, 2026-11-09), CPU.

A wire beside sand or dough, and a wire dropped on a paper or metal sheet, built and then raised SolverVBD's own
NotImplementedError at the first step, with no pair named and no fix. A wire on a soft pad or on cloth stepped, and
then ``sim.state()`` raised IndexError (the wire's plastic-strain read overwrote the sheets' one in
``alloy/api/_state.py``). Now:
- rod_vbd beside an MPM material is refused at build (``alloy.coupling.unsupported.UNSUPPORTED``);
- rod_vbd beside a film sheet is refused at build when the pair runs on the proxy coupler (``ROD_BESIDE_FILM``); a
  pair strong enough for the ADMM coupler steps, as before;
- every refusal names both objects and the way out; the state of a wire beside cloth or a soft body reads.
What works (docs/guide/rods.md, Robots): a rod_vbd rod beside rigid bodies and robots, a chain rod, cloth, a soft
body and a liquid.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import alloy
from alloy.coupling import unsupported as U
from alloy.coupling.types import PairPolicy
from alloy.route.types import Route

SHEETS = {"paper": {"material": "paper_80gsm", "thickness": 0.002},
          "metal": {"material": "aluminium_6061_t6", "thickness": 0.0005}}


def _wire(sc: alloy.Scene, top: float, n: float = 0.16) -> None:
    sc.add_rod("wire", length=n, radius=0.0015, pos=(0, -n / 2, top + 0.0315), direction=(0, 1, 0),
               material="copper_c110")


def _scene() -> alloy.Scene:
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    return sc


INTERNAL = ("SolverVBD", "NotImplementedError", "ADMM", "proxy", "sweep", "COUPLE-2", "vbd:", "mpm:", "film:", "SCR-")


def _plain(e: alloy.BuildError) -> None:
    """The refusal names objects, kinds and routes, not engine classes, couplers or ticket ids (audit re-test)."""
    text = f"{e.message} {e.fix}".replace("UnsupportedPairError", "")
    assert not [w for w in INTERNAL if w in text], text


@pytest.mark.parametrize("material", ["sand_silica_dry", "dough_wheat"])
def test_a_wire_beside_sand_or_dough_is_refused_at_build(material):
    sc = _scene()
    sc.add_open_box("tray", inner=(0.12, 0.12, 0.04), wall=0.006, pos=(0, 0, 0))
    sc.add_box("bed", size=(0.1, 0.1, 0.02), pos=(0, 0, 0.017), material=material)
    _wire(sc, 0.03, 0.1)
    with pytest.raises(alloy.BuildError) as err:
        alloy.build(sc, device="cpu")
    e = err.value
    assert "wire (elastic rod) on route rod_vbd" in e.message and "bed (" in e.message
    assert "UnsupportedPairError" in e.message
    assert "rigid=True" in e.fix and "chain route" in e.fix
    _plain(e)


@pytest.mark.parametrize("kind", sorted(SHEETS))
def test_a_wire_dropped_on_a_film_sheet_is_refused_at_build(kind):
    """The scene is the rod and the film sheet alone (two entries, plus a static stand): the assembler lists their
    pair in ``built.coupler.pairs``, which the check reads, so no default pair is made after it (WP12's review)."""
    sc = _scene()
    sc.add_box("stand", size=(0.3, 0.3, 0.2), pos=(0, 0, 0.1), static=True)
    sc.add_cloth("sheet", size=(0.3, 0.3), pos=(0, 0, 0.204), resolution=20, **SHEETS[kind])
    _wire(sc, 0.2)
    with pytest.raises(alloy.BuildError) as err:
        alloy.build(sc, device="cpu")
    e = err.value
    assert "wire (elastic rod) on route rod_vbd and sheet (film sheet) on route cloth_film" in e.message
    assert "chain route" in e.fix and "cotton_woven" in e.fix
    _plain(e)


def test_the_film_refusal_follows_the_coupler():
    """Refused on a lagged or staggered pair only: on ADMM the rod and the film sheet step (a clamped wire beside a
    paper sheet did, 5 periods, check ok)."""
    d = {"wire": SimpleNamespace(route=Route.ROD_VBD, entry="vbd"),
         "sheet": SimpleNamespace(route=Route.CLOTH_FILM, entry="film")}

    def built(mode: str) -> SimpleNamespace:
        return SimpleNamespace(report=SimpleNamespace(decisions=d),
                               coupler=SimpleNamespace(pairs=(PairPolicy("film", "vbd", mode=mode),)))

    for mode in ("lagged", "staggered"):
        with pytest.raises(U.UnsupportedPairError) as err:
            U.check_built(built(mode))
        assert err.value.pair == (Route.ROD_VBD, Route.CLOTH_FILM) and err.value.fix == U.ROD_BESIDE_FILM.alternative
    U.check_built(built("admm"))
    assert U.unsupported(Route.ROD_VBD, Route.GRAIN_MPM) is not None
    for ok in (Route.RIGID, Route.CHAIN, Route.CLOTH_VBD, Route.SOFT_VBD, Route.LIQUID_SPH, Route.CLOTH_FILM):
        assert U.unsupported(Route.ROD_VBD, ok) is None, ok


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
@pytest.mark.parametrize("kind", ["cloth", "soft"])
def test_the_state_of_a_wire_on_cloth_or_a_soft_pad_reads(kind):
    sc = _scene()
    if kind == "cloth":
        sc.add_box("stand", size=(0.3, 0.3, 0.2), pos=(0, 0, 0.1), static=True)
        sc.add_cloth("sheet", size=(0.3, 0.3), pos=(0, 0, 0.204), resolution=10, material="cotton_woven")
        other, top = "sheet", 0.2
    else:
        sc.add_box("pad", size=(0.2, 0.2, 0.04), pos=(0, 0, 0.021), material="silicone_40a")
        other, top = "pad", 0.04
    _wire(sc, top)
    sim = alloy.build(sc, device="cpu")
    assert sim.routes()["wire"]["route"] == "rod_vbd"
    sim.step(2)
    st = sim.state(world=0)                                    # raised IndexError (index 1, axis 0 with size 1)
    assert np.isfinite(st["wire.points"]).all() and np.isfinite(st[f"{other}.points"]).all()
    assert st["wire.plastic_strain"].ndim == 1 and f"{other}.plastic_strain" not in st
    sim.close()
