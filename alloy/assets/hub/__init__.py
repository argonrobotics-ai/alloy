# SPDX-License-Identifier: Apache-2.0
"""Alloy's object library on the Hugging Face Hub: lazy per-object download into Alloy's asset cache.

The dataset (``alloy-library``) holds every licence-clean object: card, material card, meshes, hull sidecars,
textures, looks and thumbnail, one folder per object, with ``index.jsonl`` listing them (``layout``). ``fetch`` puts
objects into ``$ALLOY_CACHE/imports/<library>/`` (default ``~/.cache/alloy/imports/<library>/``), where Alloy's card
libraries find them.
"""
