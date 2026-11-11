# SPDX-License-Identifier: Apache-2.0
"""Material presets for the Cycles renderer (pure data, no bpy).

Colours are sRGB 0-1. Lengths are metres. Keys beyond the look-file overrides in docs/RENDER.md:

- ``hard``: hard surface, gets a Bevel shader node of ``bevel_m``.
- ``subsurface_rgb``: relative scattering distance per channel (times ``subsurface_radius_m``).
- ``rough_var``: relative roughness variation from low-frequency noise (nothing looks CG-perfect).
- ``tint_var``: relative base-colour variation from noise.
- ``micro_bump_m``: height of a fine procedural bump.
- ``detail``: a procedural detail recipe (``brushed``, ``crust``, ``crumb``, ``fibre``, ``plating``, ``scratches``).
- ``texture`` / ``scale_m`` / ``bump_m`` / ``res``: a CC0 texture set mapped triplanar, its tile size, its height
  range and its download resolution. A texture on any preset replaces ``base_color`` and keeps every other value
  (``roughness`` given in the look overrides the texture's roughness map). ``tex_tint`` instead uses only the
  texture's luminance pattern around ``base_color`` (a blue terry-cloth scan becomes a towel of any colour).
"""

from __future__ import annotations

OVERRIDE_KEYS = (
    "base_color", "roughness", "metallic", "transmission", "ior", "subsurface", "subsurface_radius_m", "sheen", "coat",
    "anisotropic", "absorption_color", "absorption_m", "emission",
)
EXTRA_KEYS = (
    "texture", "scale_m", "bump_m", "res", "hard", "bevel_m", "rough_var", "tint_var", "micro_bump_m", "detail",
    "coat_roughness", "sheen_roughness", "anisotropic_rotation", "specular", "alpha", "mapping", "preset",
    "subsurface_color", "roughness_gain", "tex_tint", "subsurface_rgb", "texture_alpha", "emission_strength",
    "keys", "imperfection", "imperfection_scale_m", "imperfection_strength", "interp", "color_gain",
    "planar_axis", "planar_origin", "planar_up",
)

DEFAULTS: dict = {
    "base_color": [0.8, 0.8, 0.8], "roughness": 0.5, "metallic": 0.0, "transmission": 0.0, "ior": 1.5,
    "subsurface": 0.0, "subsurface_radius_m": 0.001, "sheen": 0.0, "sheen_roughness": 0.5, "coat": 0.0,
    "coat_roughness": 0.05, "anisotropic": 0.0, "anisotropic_rotation": 0.0, "specular": 0.5, "emission": None,
    "absorption_color": None, "absorption_m": None, "alpha": 1.0, "hard": False, "bevel_m": 0.0005,
    "rough_var": 0.12, "tint_var": 0.03, "micro_bump_m": 0.0, "detail": None, "texture": None, "scale_m": 0.5,
    "bump_m": 0.0003, "res": "2k", "mapping": "triplanar", "roughness_gain": 1.0, "tex_tint": False,
    "subsurface_rgb": [1.0, 0.55, 0.3], "texture_alpha": False, "emission_strength": None, "keys": None,
    "interp": "constant", "imperfection": None, "imperfection_scale_m": 0.1, "imperfection_strength": 0.3,
    "color_gain": 1.0, "planar_axis": [0.0, 0.0, 1.0], "planar_origin": [0.0, 0.0, 0.0], "planar_up": None,
}

_HARD = {"hard": True}
_METAL = {"metallic": 1.0, "hard": True, "rough_var": 0.2}

PRESETS: dict[str, dict] = {
    # Plastics
    "plastic_matte": {**_HARD, "base_color": [0.6, 0.6, 0.6], "roughness": 0.45, "ior": 1.46,
                      "micro_bump_m": 0.00001},
    "plastic_gloss": {**_HARD, "base_color": [0.6, 0.6, 0.6], "roughness": 0.12, "ior": 1.49, "rough_var": 0.25},
    "plastic_clear": {**_HARD, "base_color": [1.0, 1.0, 1.0], "roughness": 0.04, "transmission": 1.0, "ior": 1.585,
                      "absorption_color": [0.96, 0.97, 0.98], "absorption_m": 0.01},
    "polycarbonate_clear": {**_HARD, "base_color": [1.0, 1.0, 1.0], "roughness": 0.02, "transmission": 1.0,
                            "ior": 1.585, "absorption_color": [0.97, 0.98, 0.99], "absorption_m": 0.02,
                            "rough_var": 0.25, "micro_bump_m": 0.000002, "tint_var": 0.0, "bevel_m": 0.0002},
    "pvc_jacket_matte": {"base_color": [0.3, 0.36, 0.45], "roughness": 0.55, "ior": 1.54, "sheen": 0.1,
                         "rough_var": 0.15, "micro_bump_m": 0.00001},
    "abs_switch_housing": {**_HARD, "base_color": [0.11, 0.11, 0.12], "roughness": 0.45, "ior": 1.53,
                           "micro_bump_m": 0.00002, "rough_var": 0.1},
    "led_green": {**_HARD, "base_color": [0.06, 0.32, 0.1], "roughness": 0.05, "coat": 1.0, "coat_roughness": 0.02,
                  "subsurface": 0.3, "subsurface_radius_m": 0.001, "emission": [0.15, 1.0, 0.25],
                  "emission_strength": 6.0, "rough_var": 0.0, "tint_var": 0.0},
    "led_amber": {**_HARD, "base_color": [0.45, 0.25, 0.03], "roughness": 0.05, "coat": 1.0, "coat_roughness": 0.02,
                  "subsurface": 0.3, "subsurface_radius_m": 0.001, "emission": [1.0, 0.55, 0.05],
                  "emission_strength": 6.0, "rough_var": 0.0, "tint_var": 0.0},
    "pp_clear": {**_HARD, "base_color": [1.0, 1.0, 1.0], "roughness": 0.2, "transmission": 0.9, "ior": 1.49,
                 "absorption_color": [0.97, 0.95, 0.9], "absorption_m": 0.005, "rough_var": 0.3,
                 "detail": "scratches", "micro_bump_m": 0.000003, "tint_var": 0.0},
    "pp_frosted": {**_HARD, "base_color": [0.95, 0.95, 0.94], "roughness": 0.45, "transmission": 0.85, "ior": 1.49,
                   "subsurface": 0.15, "subsurface_radius_m": 0.001, "rough_var": 0.15, "micro_bump_m": 0.000004,
                   "tint_var": 0.0},
    "plastic_translucent_white": {**_HARD, "base_color": [0.9, 0.9, 0.88], "roughness": 0.3, "subsurface": 0.6,
                                  "subsurface_radius_m": 0.004},
    "nylon_natural": {**_HARD, "base_color": [0.9, 0.88, 0.8], "roughness": 0.42, "ior": 1.53, "subsurface": 0.35,
                      "subsurface_radius_m": 0.0025, "micro_bump_m": 0.00002},
    "abs_black": {**_HARD, "base_color": [0.035, 0.035, 0.038], "roughness": 0.33, "ior": 1.53,
                  "micro_bump_m": 0.00001},
    "rubber_black": {**_HARD, "base_color": [0.028, 0.028, 0.028], "roughness": 0.72, "sheen": 0.15,
                     "rough_var": 0.15, "micro_bump_m": 0.00002},
    "rubber_grey": {**_HARD, "base_color": [0.25, 0.25, 0.26], "roughness": 0.7, "sheen": 0.1},
    "pvc_cable_black": {"base_color": [0.022, 0.022, 0.024], "roughness": 0.32, "ior": 1.54, "rough_var": 0.2,
                        "coat": 0.2, "coat_roughness": 0.15},
    "pvc_cable_grey": {"base_color": [0.46, 0.47, 0.48], "roughness": 0.36, "ior": 1.54, "rough_var": 0.2},
    "robot_painted_white": {**_HARD, "base_color": [0.86, 0.87, 0.87], "roughness": 0.28, "coat": 0.5,
                            "coat_roughness": 0.09, "micro_bump_m": 0.000006},
    "robot_painted_dark": {**_HARD, "base_color": [0.06, 0.065, 0.07], "roughness": 0.3, "coat": 0.5,
                           "coat_roughness": 0.1, "micro_bump_m": 0.000006},
    "powder_coat_black": {**_HARD, "base_color": [0.04, 0.04, 0.042], "roughness": 0.55, "micro_bump_m": 0.00003},
    "pcb_green": {**_HARD, "base_color": [0.05, 0.24, 0.1], "roughness": 0.22, "coat": 1.0, "coat_roughness": 0.12,
                  "subsurface": 0.1, "subsurface_radius_m": 0.0005},
    # Metals
    "steel_brushed": {**_METAL, "base_color": [0.77, 0.77, 0.78], "roughness": 0.28, "anisotropic": 0.75,
                      "detail": "brushed", "micro_bump_m": 0.000004},
    "steel_polished": {**_METAL, "base_color": [0.78, 0.78, 0.79], "roughness": 0.06},
    "steel_black_oxide": {**_METAL, "base_color": [0.12, 0.12, 0.13], "roughness": 0.35},
    "zinc_plated": {**_METAL, "base_color": [0.83, 0.84, 0.87], "roughness": 0.24, "rough_var": 0.35,
                    "detail": "plating"},
    "brass": {**_METAL, "base_color": [0.95, 0.82, 0.52], "roughness": 0.18},
    "gold_plated": {**_METAL, "base_color": [1.0, 0.83, 0.47], "roughness": 0.1, "anisotropic": 0.4,
                    "detail": "brushed", "micro_bump_m": 0.000002},
    "copper_wire": {**_METAL, "base_color": [0.96, 0.64, 0.54], "roughness": 0.22, "anisotropic": 0.5,
                    "detail": "brushed", "hard": False},
    "aluminum_bare": {**_METAL, "base_color": [0.91, 0.92, 0.92], "roughness": 0.3, "anisotropic": 0.35,
                      "detail": "brushed"},
    "aluminum_anodized_black": {**_HARD, "base_color": [0.035, 0.035, 0.04], "metallic": 0.5, "roughness": 0.32,
                                "rough_var": 0.15, "micro_bump_m": 0.000008},
    "copper": {**_METAL, "base_color": [0.96, 0.64, 0.54], "roughness": 0.17},
    # Glass and liquids (Volume Absorption inside a transmissive surface)
    "glass_clear": {**_HARD, "base_color": [1.0, 1.0, 1.0], "roughness": 0.0, "transmission": 1.0, "ior": 1.52,
                    "absorption_color": [0.88, 0.96, 0.93], "absorption_m": 0.1, "rough_var": 0.0, "tint_var": 0.0},
    "water": {"base_color": [1.0, 1.0, 1.0], "roughness": 0.005, "transmission": 1.0, "ior": 1.333,
              "absorption_color": [0.9, 0.97, 0.98], "absorption_m": 1.0, "rough_var": 0.0, "tint_var": 0.0},
    "coolant": {"base_color": [1.0, 1.0, 1.0], "roughness": 0.005, "transmission": 1.0, "ior": 1.36,
                "absorption_color": [0.35, 0.75, 0.95], "absorption_m": 0.02, "rough_var": 0.0, "tint_var": 0.0},
    "honey": {"base_color": [1.0, 1.0, 1.0], "roughness": 0.015, "transmission": 1.0, "ior": 1.49,
              "absorption_color": [0.9, 0.52, 0.08], "absorption_m": 0.02, "rough_var": 0.0, "tint_var": 0.0},
    "tea": {"base_color": [1.0, 1.0, 1.0], "roughness": 0.005, "transmission": 1.0, "ior": 1.335,
            "absorption_color": [0.62, 0.26, 0.08], "absorption_m": 0.02, "rough_var": 0.0, "tint_var": 0.0},
    # Ceramic, stone, wood
    "ceramic_white": {**_HARD, "base_color": [0.9, 0.89, 0.86], "roughness": 0.35, "coat": 1.0,
                      "coat_roughness": 0.03, "subsurface": 0.1, "subsurface_radius_m": 0.001},
    "marble_white": {**_HARD, "base_color": [1.0, 1.0, 1.0], "roughness": 0.18, "subsurface": 0.15,
                     "subsurface_radius_m": 0.004, "texture": "polyhaven:marble_01", "scale_m": 0.8,
                     "bump_m": 0.00005},
    "wood_oak": {**_HARD, "base_color": [1.0, 1.0, 1.0], "roughness": 0.5, "texture": "polyhaven:oak_veneer_01",
                 "scale_m": 0.6, "bump_m": 0.0002, "rough_var": 0.1},
    "wood_walnut_oiled": {**_HARD, "base_color": [1.0, 1.0, 1.0], "roughness": 0.32, "coat": 0.35,
                          "coat_roughness": 0.18, "texture": "polyhaven:american_walnut_veneer", "scale_m": 0.6,
                          "bump_m": 0.00015, "rough_var": 0.15},
    "wood_plywood": {**_HARD, "base_color": [1.0, 1.0, 1.0], "roughness": 0.62, "texture": "polyhaven:plywood",
                     "scale_m": 0.8, "bump_m": 0.0001, "rough_var": 0.1},
    "plaster_paint": {"base_color": [0.8, 0.79, 0.76], "roughness": 0.88, "micro_bump_m": 0.00006, "tint_var": 0.02,
                      "rough_var": 0.05},
    "grey_card_18": {"base_color": [0.4613, 0.4613, 0.4613], "roughness": 0.9, "rough_var": 0.0, "tint_var": 0.0},
    # Textiles and paper
    "linen": {"base_color": [0.86, 0.83, 0.76], "roughness": 0.85, "sheen": 0.6, "sheen_roughness": 0.4,
              "texture": "polyhaven:rough_linen", "scale_m": 0.15, "bump_m": 0.0004, "subsurface": 0.05,
              "tex_tint": True},
    "cotton_terry": {"base_color": [0.92, 0.91, 0.88], "roughness": 0.9, "sheen": 1.0, "sheen_roughness": 0.35,
                     "texture": "polyhaven:terry_cloth", "scale_m": 0.12, "bump_m": 0.0012, "subsurface": 0.1,
                     "subsurface_radius_m": 0.002, "tex_tint": True},
    "cardboard_kraft": {"base_color": [1.0, 1.0, 1.0], "roughness": 0.85, "texture": "ambientcg:Cardboard004",
                        "scale_m": 0.4, "bump_m": 0.0002, "sheen": 0.1},
    "paper": {"base_color": [0.93, 0.93, 0.91], "roughness": 0.75, "subsurface": 0.2, "subsurface_radius_m": 0.0006,
              "sheen": 0.1, "micro_bump_m": 0.00003, "detail": "fibre"},
    # Food
    "bread_crust": {"base_color": [0.66, 0.42, 0.2], "roughness": 0.5, "subsurface": 0.08,
                    "subsurface_radius_m": 0.002, "detail": "crust", "micro_bump_m": 0.0008, "tint_var": 0.18,
                    "rough_var": 0.25},
    "bread_crumb": {"base_color": [0.94, 0.86, 0.68], "roughness": 0.85, "subsurface": 0.45,
                    "subsurface_radius_m": 0.003, "detail": "crumb", "micro_bump_m": 0.0012, "tint_var": 0.06},
    "dough": {"base_color": [0.95, 0.88, 0.72], "roughness": 0.5, "subsurface": 0.6, "subsurface_radius_m": 0.005,
              "micro_bump_m": 0.0001, "tint_var": 0.04},
    "rice_grain": {"base_color": [0.88, 0.85, 0.76], "roughness": 0.3, "subsurface": 1.0,
                   "subsurface_radius_m": 0.003, "subsurface_rgb": [1.0, 0.9, 0.7], "tint_var": 0.08, "ior": 1.53,
                   "coat": 0.25, "coat_roughness": 0.2},
}

GRAINS: dict[str, dict] = {
    # Semi-axes in metres before the look's ``scale``; ``taper`` narrows the grain toward its tips.
    "rice_grain": {"axes": [0.0032, 0.00105, 0.0009], "taper": 0.25, "segments": 16, "rings": 10},
    "sand_grain": {"axes": [0.0003, 0.00025, 0.0002], "taper": 0.0, "segments": 8, "rings": 6, "lumpy": 0.25},
    "sphere": {"axes": [1.0, 1.0, 1.0], "taper": 0.0, "segments": 16, "rings": 8, "unit": True},
}

FALLBACK_MATERIAL = "plastic_matte"

# Aliases: same values under another name.
PRESETS["stainless_steel_brushed"] = PRESETS["steel_brushed"]
