# SPDX-License-Identifier: Apache-2.0
"""Wavefront path tracing: Warp shades, a tracer casts. ``RenderSettings(backend="optix")`` casts on RT cores
(:mod:`alloy.camera.optix`); ``"wavefront-warp"`` casts with Warp's BVH (validation: the wavefront split itself must
reproduce the megakernel to float rounding).

* :mod:`.host`: buffers, waves, the per-bounce loop (``Wavefront``, ``Stages``)
* :mod:`.diffuse`: the diffuse integrator; :mod:`.pbr`, :mod:`.pbr_stages`, :mod:`.pbr_shade`: the full-material one
* :mod:`.bevel`, :mod:`.bevel_flat`, :mod:`.sort`: Bevel as a pass, its flat-region early-out, the material sort
* :mod:`.knobs`: the ``ALLOY_CAMERA_WF_*`` environment knobs (``WavefrontOptions``)

The renderers pick the wavefront themselves from ``settings.backend``; this package is not used directly.
"""
from .knobs import WavefrontOptions

__all__ = ["WavefrontOptions"]
