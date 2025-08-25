# utils/standardize.py
# --------------------------------------------------------------------
# Helpers to standardize system_type labels for nodes based on a
# shared taxonomy with synonyms. This ensures consistency across
# communities (e.g., "PD", "Sheriff", "Police Dept." → "Law Enforcement").
# --------------------------------------------------------------------

from __future__ import annotations
from typing import List, Dict


def standardize_system_type(
    raw: str,
    system_types: List[str],
    synonyms: Dict[str, List[str]],
) -> str:
    """
    Standardize a raw system_type label into one of the canonical system_types.

    Parameters
    ----------
    raw : str
        The raw label from partner input (e.g., "PD", "Hospital").
    system_types : List[str]
        Canonical list of system types from taxonomy.json.
    synonyms : Dict[str, List[str]]
        Map of canonical system type → list of synonyms.

    Returns
    -------
    str : standardized system type, or "Other" if no match found.
    """
    if not raw:
        return "Other"

    s = str(raw).strip().lower()

    # Check direct match to canonical types
    for t in system_types:
        if s == t.lower():
            return t

    # Check synonyms dictionary
    for canon, syns in (synonyms or {}).items():
        if s in [ss.lower() for ss in syns]:
            return canon

    # Fallback: attempt loose matching (e.g., substring search)
    for t in system_types:
        if s in t.lower() or t.lower() in s:
            return t

    return "Other"
