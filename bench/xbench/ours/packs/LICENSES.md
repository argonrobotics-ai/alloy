# Licences of the packs in this directory

The `emb_*` packs are exports of the robot models in `library/robots/emb_*` (made with
`bench/xbench/ours/scene_pack.py`). Each keeps the licence of its source model; the licence texts are shipped as
`library/robots/<same name>/LICENSE.<n>` and listed in the top-level `NOTICE`:

| Packs | Source | Licence |
|---|---|---|
| `emb_e1_fr3`, `emb_e2_panda`, `emb_e4_fr3_bimanual*` | MuJoCo Menagerie franka_fr3, franka_emika_panda | Apache-2.0 |
| `emb_e3_ur10e_2f85*` | MuJoCo Menagerie universal_robots_ur10e, robotiq_2f85 | BSD-3-Clause, BSD-2-Clause |
| `emb_e5_fr3_duo*` | franka_description (gains and pose: gym-anything, MIT) | Apache-2.0 |
| `emb_e6_g1*`, `emb_e8_go2*` | MuJoCo Menagerie unitree_g1, unitree_go2 | BSD-3-Clause |
| `emb_e7_h1_2*` | unitree_ros h1_2_description (gains: gym-anything, MIT) | BSD-3-Clause |
| `emb_e9_leap` | MuJoCo Menagerie leap_hand | MIT |

The other packs (`laws/`, `t12/`) are Alloy's own (Apache-2.0).
