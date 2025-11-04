"""
Stubs for legacy web search utilities referenced by voice evaluators.

The project no longer exposes live web search capabilities, but older releases
still import ``utils.web_search.is_browsecomp_episode`` to decide whether the
BrowseComp tooling should run.  We keep a minimal shim so those imports resolve
without pulling in unavailable dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def is_browsecomp_episode(episode_data: Optional[Dict[str, Any]]) -> bool:
    """Return True if this episode should enable web search tooling.

    Strategy: enable only for browsecomp benchmark by id/track hints.
    """
    episode_id = (episode_data or {}).get("id", "").lower()
    track = (episode_data or {}).get("track", "").lower()
    return "browsecomp" in episode_id or track == "browsecomp"
