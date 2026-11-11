# SPDX-License-Identifier: Apache-2.0
"""Law of the proxy's inverse-mass tensor in VBD: phase P1 of known gap
docs/known_gaps/coupling-proxy-isotropic-mass.md (notes: [internal note] notes/wp10-aniso/law_p1.md).

A MuJoCo body coupled into VBD is a proxy there with one scalar mass, 3 / trace(W), W the body's translational block
of J M^-1 J^T at its centre of mass at qpos0. The law: the proxy answers a force as its body does, Minv = W, along
the body's moving axis and across it. Rig ``proxy_tensor_rig``: C3's 0.515 kg platen over its free 16 g pad, mid-air,
one MJCF body whose joint alone changes.

**How the answer is read.** The coupler applies each exchange's feedback force f to the MuJoCo body, then takes the
same f back from the proxy inside VBD (the lagged rewind) so that VBD's contact solve does not count it twice. A known
f in the first pass of an exchange, in free flight, is answered by MuJoCo with dt W f and by VBD's step with
-dt Minv f; the second pass repeats the exchange without it. The difference of the passes is each solver's answer to
f alone: no contact model, no VBD convergence, no gravity. Today (CPU Warp route, 2026-11-09, 0.156 ms exchanges,
kicks of 40 m/s^2 along the moving axis):
- slider (one vertical slide): body 1.941 /kg along z and 0 across; proxy 0.647 /kg in every direction (1/3).
- hinge (the same box on a hinge 5 cm from its centre of mass): body 1.712 /kg along z; proxy 0.571 /kg (1/3).
- free and xyz (three slides, the slider's twin): body and proxy 1.941 /kg. Exact, and they must stay so.

**In a contact.** The proxy's velocity change in a contact exchange over the impulse VBD harvests for it reads no
mass: 0.05 to 48 /kg per exchange for the exact free proxy (1.94 /kg), and 1.66 /kg summed over the impact for the
slider's 0.647 /kg. The harvest is a force at VBD's last iterate, not the impulse of its unconverged solve
(``record_momentum``'s reason). Contact is read against a twin instead: the slider and the xyz body start the same
contact from the same state, and in the first pass of the first exchange (no feedback yet) VBD must push their
proxies back alike. Today 16.1 against 65.0 mm/s. Later exchanges move with the float32 grid, so only the first is
the law; the pad's momentum over the whole contact is a control (-4.0 % today, floor 1.4 %).

The gap page's 1.2 g hinge tip does not appear: over this pad it runs SchedulerADMM, which has no proxy (rig
``coupling_class``, 2026-11-09; the page: light bodies are strong pairs). The hinge case is the 0.515 kg platen, which
couples through the proxy.
It has no contact case: no body is its exact twin in a contact (its face meets the pad at 100 +- 40 mm/s as it turns).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pytest

from tests.coupling import proxy_tensor_rig as R

pytestmark = [pytest.mark.law, pytest.mark.compile_heavy, pytest.mark.timeout(900)]

TOL = 0.10                         # of the body's answer along its moving axis
AXIS = 2                           # every kind moves along z at qpos0
G = 9.81
GAP = "known gap coupling-proxy-isotropic-mass: "
ONE_AXIS = {
    "slider": "0.647 /kg against the body's 1.941 /kg",
    "hinge": "0.571 /kg against the body's 1.712 /kg",
}


@pytest.fixture(scope="module", params=["scalar", "tensor"])
def mode(request: pytest.FixtureRequest) -> Iterator[str]:
    """Every law runs twice: with the shipped scalar proxy mass, and with the opt-in tensor
    (``ALLOY_PROXY_TENSOR=1``, read when the coupler is built). The one-axis laws fail on the scalar (the known
    gap, strict xfail) and must pass on the tensor; every control must pass on both."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ALLOY_PROXY_TENSOR", "1" if request.param == "tensor" else "0")
        yield request.param


def _known_gap(request: pytest.FixtureRequest, mode: str, reason: str) -> None:
    """Mark the running test as the known gap's strict xfail on the scalar proxy; the tensor must pass it."""
    if mode == "scalar":
        request.applymarker(pytest.mark.xfail(strict=True, reason=GAP + reason))


@pytest.fixture(scope="module")
def rewind(mode: str) -> Callable[[str], R.Rewind]:
    """Each kind's answers to a known feedback force, built and measured once."""
    cache: dict[str, R.Rewind] = {}

    def get(kind: str) -> R.Rewind:
        if kind not in cache:
            cache[kind] = R.rewind_response(R.build(kind))
        return cache[kind]

    return get


def _along(r: R.Rewind) -> float:
    """The proxy's answer over its body's, along the moving axis (target 1)."""
    return float(r.proxy[AXIS, AXIS] / r.source[AXIS, AXIS])


def _across(r: R.Rewind) -> float:
    """The largest answer the proxy gives and its body does not, off the moving axis, over the body's answer along
    it (target 0)."""
    extra = np.abs(r.proxy - r.source)
    extra[AXIS, AXIS] = 0.0
    return float(extra.max() / r.scale)


@pytest.mark.parametrize("kind", ["free", "xyz", "slider", "hinge"])
def test_the_rig_reads_mujocos_own_w(rewind: Callable[[str], R.Rewind], kind: str) -> None:
    """The source's measured answer is W from MuJoCo's model (mj_jacBodyCom, mj_solveM) within 1 %, the same in
    every exchange, and the proxy installed is 3 / trace(W)."""
    r = rewind(kind)
    np.testing.assert_allclose(r.source, r.w, rtol=0.0, atol=0.01 * r.scale)
    assert r.spread <= 0.01 * r.scale, r
    assert r.proxy_mass == pytest.approx(3.0 / np.trace(r.w), rel=1e-4), r
    if kind == "hinge":
        assert 1.0 / r.w[AXIS, AXIS] == pytest.approx(R.HINGE_AXIS_MASS, rel=1e-4), r     # m + I_yy / ARM^2


@pytest.mark.parametrize("kind", ["free", "xyz", "slider", "hinge"])
def test_proxy_falls_with_its_body_between_contacts(rewind: Callable[[str], R.Rewind], kind: str) -> None:
    """Without a feedback force VBD's step leaves the proxy's velocity as the source gave it: gravity is taken back
    with the mass that applies it. Passes today; a tensor in the step with the scalar in the rewind would leave
    2 g dt here."""
    r = rewind(kind)
    assert r.drift <= 0.1 * G * r.dt, (f"{kind}: proxy drifts {r.drift * 1e3:.4f} mm/s in a step of free flight "
                                       f"(g dt = {G * r.dt * 1e3:.3f} mm/s)")


@pytest.mark.parametrize("kind", ["free", "xyz"])
def test_exact_proxy_answers_as_its_body(rewind: Callable[[str], R.Rewind], kind: str) -> None:
    """Control: W = I / m, the scalar is exact. Along and across within ``TOL`` (today 1.000 and 0.000)."""
    r = rewind(kind)
    assert abs(_along(r) - 1.0) <= TOL, f"{kind}: proxy / body along z {_along(r):.3f}, target 1"
    assert _across(r) <= TOL, f"{kind}: proxy answers {_across(r):.3f} of lambda_max(W) where its body does not"


@pytest.mark.parametrize("kind", list(ONE_AXIS))
def test_one_axis_proxy_answers_along_its_axis_as_its_body(rewind: Callable[[str], R.Rewind], kind: str, mode: str,
                                                           request: pytest.FixtureRequest) -> None:
    """A body with one moving direction: inside VBD its proxy answers a force along that direction as MuJoCo's
    dynamics do. Today 1/3 of it (slider 0.647 against 1.941 /kg, hinge 0.571 against 1.712 /kg)."""
    _known_gap(request, mode, f"the {kind}'s proxy answers a force along its axis with {ONE_AXIS[kind]} (ratio "
               f"0.333); target 1 within {TOL}")
    r = rewind(kind)
    assert abs(_along(r) - 1.0) <= TOL, (f"{kind}: proxy {r.proxy[AXIS, AXIS]:.3f} /kg, body "
                                         f"{r.source[AXIS, AXIS]:.3f} /kg along z: ratio {_along(r):.3f}, target 1 "
                                         f"within {TOL}")


@pytest.mark.parametrize("kind", list(ONE_AXIS))
def test_one_axis_proxy_stays_where_its_body_cannot_move(rewind: Callable[[str], R.Rewind], kind: str, mode: str,
                                                         request: pytest.FixtureRequest) -> None:
    """Across its moving direction the body does not answer a force, so the proxy must not either (and a force along
    the axis moves neither sideways). Today the proxy answers 1/3 of the body's along-axis answer in every
    direction. The tensor keeps a floor there (lambda_max / kappa, 0.01 of the answer along the axis at the default
    kappa 100) so that its inverse exists."""
    _known_gap(request, mode, f"across its axis the {kind}'s proxy answers with {ONE_AXIS[kind].split()[0]} /kg "
               f"where its body cannot move (0.333 of the body's answer along the axis); target 0 within {TOL}")
    r = rewind(kind)
    assert _across(r) <= TOL, (f"{kind}: across its axis the proxy answers {_across(r):.3f} of the body's "
                               f"{r.scale:.3f} /kg along it, target 0 within {TOL}")


# ---------------------------------------------------------------------------------------------- in a contact
@pytest.fixture(scope="module")
def contacts(mode: str) -> Callable[[str], R.Contact]:
    """Each kind at 0.1 m/s into the pad it already touches (two worlds, 2 control periods), run once."""
    cache: dict[str, R.Contact] = {}

    def get(kind: str) -> R.Contact:
        if kind not in cache:
            cache[kind] = R.contact(R.build(kind, worlds=2, gap=R.TOUCH))
        return cache[kind]

    return get


def _give_ratio(body: R.Contact, twin: R.Contact) -> float:
    """How far VBD's contact solve moves the body's proxy in the first pass of the first exchange, over its twin's.
    Both start from the same state and neither has a feedback force yet."""
    assert body.first_feedback == 0.0 == twin.first_feedback, (body, twin)
    assert twin.first_give > 0.01, twin                      # the pad pushes the twin's proxy back by 65 mm/s
    return body.first_give / twin.first_give


def test_slider_proxy_gives_way_in_a_contact_as_its_exact_twin(contacts: Callable[[str], R.Contact], mode: str,
                                                               request: pytest.FixtureRequest) -> None:
    """The slider and the xyz body are the same body in this contact (same mass along z, no rotation, same shape and
    start), so VBD's contact solve must move their proxies alike. This reads the mass in VBD's contact solve (the
    AVBD linear block), which the rewind answers above do not reach. Resolution floor 0.8 %: the same exchange with
    the scene 0.3 m higher on the float32 grid (slider 16.14 -> 16.27 mm/s, twin 65.00 -> 64.91)."""
    _known_gap(request, mode, "in the first contact exchange VBD pushes the slider's proxy back by 16.1 mm/s and its "
               f"exact twin's by 65.0 mm/s (ratio 0.248); target 1 within {TOL}")
    ratio = _give_ratio(contacts("slider"), contacts("xyz"))
    assert abs(ratio - 1.0) <= TOL, (f"slider proxy {contacts('slider').first_give * 1e3:.2f} mm/s, twin "
                                     f"{contacts('xyz').first_give * 1e3:.2f} mm/s: ratio {ratio:.3f}, target 1 "
                                     f"within {TOL}")


def test_free_proxy_gives_way_in_a_contact_as_the_xyz_twin(contacts: Callable[[str], R.Contact]) -> None:
    """Control: two exact proxies of the same body (free, and three slides) are pushed back alike in the first
    contact exchange (today 64.997 and 64.998 mm/s)."""
    ratio = _give_ratio(contacts("free"), contacts("xyz"))
    assert abs(ratio - 1.0) <= TOL, f"free proxy over its xyz twin {ratio:.3f}, target 1 within {TOL}"


def test_slider_pad_takes_what_its_twins_takes(contacts: Callable[[str], R.Contact]) -> None:
    """Control, end to end: over the whole contact the pad under the slider takes the momentum the pad under its
    twin takes, and each pair's momentum closes. Today 2.275 against 2.369 mN s (-4.0 %), inside ``TOL`` already:
    the feedback is what the particles took (``record_momentum``), whatever the proxy weighs. It cannot be the red
    law (the same xyz scene 0.3 m higher differs by 1.4 %, and from a 1.7 mm flight by 5.3 %); it must stay green."""
    slider, twin = contacts("slider"), contacts("xyz")
    for c in (slider, twin):
        assert c.ok, c
        assert abs(c.pad_gain + c.platen_loss) <= 0.05 * abs(c.pad_gain), c
    share = slider.pad_gain / twin.pad_gain
    assert abs(share - 1.0) <= TOL, (f"pad under the slider {slider.pad_gain * 1e3:.3f} mN s, under its twin "
                                     f"{twin.pad_gain * 1e3:.3f} mN s: ratio {share:.3f}, target 1 within {TOL}")
