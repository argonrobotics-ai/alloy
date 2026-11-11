# SPDX-License-Identifier: Apache-2.0
"""Plug-in templates: ``python -m alloy.plugins new <name> --kind <kind>`` copies one of these modules.

Each module (``force``, ``sensor``, ``object``, ``solver``) is a complete plug-in that passes the harness unchanged;
``PluginName`` and ``plugin_name`` are replaced by the new plug-in's names. They live inside the package so they ship
in every wheel (``alloy.plugins.scaffold`` reads them as package resources).
"""
