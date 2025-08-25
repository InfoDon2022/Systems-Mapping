# utils/vignettes.py
# --------------------------------------------------------------------
# Vignette utilities for the OVW Gap Mapper.
# - find_node_by_label_or_id: resolves a node either by exact ID or by
#   case-insensitive match to its "label" attribute.
# - simulate_vignette: walks a vignette's journey steps against the
#   current graph and reports successes and gaps (missing nodes/edges).
# --------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import networkx as nx


def find_node_by_label_or_id(G: nx.Graph, label_or_id: str) -> Optional[str]:
    """
    Resolve a node in G either by node_id or by matching the node's 'label' attribute.

    Returns the node_id if found; otherwise None.
    """
    if label_or_id is None:
        return None

    # Direct ID hit
    if label_or_id in G.nodes:
        return label_or_id

    # Match by label (case-insensitive, trimmed)
    target = str(label_or_id).strip().lower()
    for n, data in G.nodes(data=True):
        lbl = str(data.get("label", "")).strip().lower()
        if lbl == target:
            return n

    return None


def simulate_vignette(G: nx.DiGraph, vignette: Dict[str, Any]) -> Dict[str, Any]:
    """
    Walk a vignette 'journey' against the graph and report status.

    Expected vignette structure (minimum):
      {
        "id": "vignette_001",
        "title": "Some scenario",
        "journey": [
          {"step": 1, "from": "start", "to_node_label_or_id": "Some Node Label"},
          {"step": 2, "from_node_id_or_label": "NODE-123", "to_node_label_or_id": "Other Node"},
          {"step": 3, "gap_check": "custom_rule_name"}   # optional placeholder
        ]
      }

    Rules:
    - If "from" == "start", we just note the first hop destination.
    - Otherwise we resolve 'from_node_id_or_label' and 'to_node_label_or_id'
      against the graph (by ID or label).
    - If the destination node is missing, record a "missing_destination" gap.
    - If the source node is missing (and not "start"), record "missing_source".
    - If both exist but there is no edge from->to, record "no_edge".
    - Any object with 'gap_check' is passed through as a placeholder finding.

    Returns:
      {
        "steps": [ ... per-step status ... ],
        "gaps":  [ ... summarized gaps ... ]
      }
    """
    steps = vignette.get("journey", []) or []
    steps_report: List[Dict[str, Any]] = []
    gaps: List[Dict[str, Any]] = []

    for step in steps:
        # Allow an explicit "gap_check" placeholder step
        if step.get("gap_check"):
            gc = str(step["gap_check"])
            steps_report.append({
                "step": step.get("step"),
                "status": "gap_check_placeholder",
                "detail": gc
            })
            gaps.append({"type": "gap_check", "what": gc})
            continue

        # Resolve endpoints
        frm_label = step.get("from_node_id_or_label") or step.get("from")
        to_label = step.get("to_node_label_or_id")

        # Destination must exist
        to_id = find_node_by_label_or_id(G, to_label) if to_label else None
        if to_id is None:
            steps_report.append({
                "step": step.get("step"),
                "status": "missing_destination",
                "from": frm_label,
                "to": to_label
            })
            gaps.append({"type": "missing_destination", "from": frm_label, "to": to_label})
            continue

        # First hop from the abstract "start"
        if frm_label == "start":
            steps_report.append({
                "step": step.get("step"),
                "status": "start_to_first",
                "to": to_id
            })
            continue

        # Source must exist (unless "start")
        frm_id = find_node_by_label_or_id(G, frm_label) if frm_label else None
        if frm_id is None:
            steps_report.append({
                "step": step.get("step"),
                "status": "missing_source",
                "to": to_id
            })
            gaps.append({"type": "missing_source", "to": to_id})
            continue

        # Check the edge
        if G.has_edge(frm_id, to_id):
            edge_data = G.get_edge_data(frm_id, to_id)
            steps_report.append({
                "step": step.get("step"),
                "status": "ok",
                "from": frm_id,
                "to": to_id,
                "edge": edge_data
            })
        else:
            steps_report.append({
                "step": step.get("step"),
                "status": "no_edge",
                "from": frm_id,
                "to": to_id
            })
            gaps.append({"type": "no_edge", "from": frm_id, "to": to_id})

    return {"steps": steps_report, "gaps": gaps}
