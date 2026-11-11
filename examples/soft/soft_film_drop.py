# SPDX-License-Identifier: Apache-2.0
"""Drop a library film or cloth on the floor in one line: plastic film sheets (film_hdpe_sheet, film_ldpe_sheet,
film_lldpe_sheet), an HDPE grocery bag (bag_hdpe_tshirt), paper sheets (paper_kraft_sheet, paper_wrap_sheet,
paper_board_sheet), cotton terry towels (towel_cotton, towel_cotton_hand) and cloth sheets (cloth_cotton_sheet,
cloth_polyester_sheet).

The line that matters: ``scene.add_object("sheet", "film_hdpe_sheet", pos=(0, 0, 0.05))``. The card brings the
material, thickness and solver route (films and paper use the thin-film solver, cloths the cloth solver).

Run: python examples/soft/soft_film_drop.py [card] [seconds]
"""
import json
import sys

import numpy as np

import alloy


def drop(card: str = "film_hdpe_sheet", seconds: float = 1.0, device: str | None = None) -> dict:
    """Drop ``card`` from 5 cm onto the ground; return its route, lowest and highest point and the health verdict."""
    scene = alloy.Scene()
    scene.add_ground()
    scene.add_object("sheet", card, pos=(0, 0, 0.05))
    sim = alloy.build(scene, device=device)
    sim.run(seconds)
    points = np.asarray(sim.state("sheet")["sheet.points"])[0]
    return {"card": card, "route": sim.routes()["sheet"]["route"], "t": round(sim.t, 3),
            "z_min_m": float(points[:, 2].min()), "z_max_m": float(points[:, 2].max()), "health_ok": sim.health()["ok"]}


if __name__ == "__main__":
    card = sys.argv[1] if len(sys.argv) > 1 else "film_hdpe_sheet"
    print(json.dumps(drop(card, float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)))
