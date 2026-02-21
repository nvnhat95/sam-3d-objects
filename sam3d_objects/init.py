"""
Package initialization hook.

Upstream versions of this project import `sam3d_objects.init` from
`sam3d_objects/__init__.py` to perform optional heavyweight initialization
(e.g. extension setup, environment tweaks, registrations).

Some forks/configurations don't require any of that. This module exists to keep
imports stable so Hydra can locate pipeline classes during instantiation.
"""

# Intentionally left minimal/no-op.

