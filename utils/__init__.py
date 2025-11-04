"""
Utility helpers for the VERA benchmark.

The voice evaluators expect a handful of shared helpers under ``utils``.
Historically these lived out-of-tree, which meant the packaged release was
missing the module entirely.  This package re-introduces the helpers so that
legacy imports like ``from utils.web_search import is_browsecomp_episode`` resolve
at runtime.
"""

from .web_search import is_browsecomp_episode

__all__ = ["is_browsecomp_episode"]
