# build_graph.py
# Core graph builder and metrics for the OVW Gap Mapper.
# - Loads taxonomy & settings JSON
# - Builds a directed NetworkX graph from Nodes/Edges tables
# - Standardizes system_type using synonyms
# - Computes common network metrics

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import json
import pandas as pd
import networkx as nx

from utils.standardize import standardize_system_type


def load_taxonomy(path: Path) -> Dict[str, Any]:
    """
    Load taxonomy JSON (system_types, synonyms, relationship_types).
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_settings(path: Path) -> Dict[str, Any]:
    """
    Load settings JSON (thresholds, visualization prefs, etc.).
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    taxonomy: Dict[str, Any],
) -> nx.DiGraph:
    """
    Build a directed graph from Nodes and Edges dataframes.

    nodes_df expected columns (minimum):
        - node_id, label, system_type
      Optional but useful:
        - latitude, longitude, services_offered, is_rural, etc.

    edges_df expected columns (minimum):
        - from_id, to_id
      Optional:
        - relationship_type, directionality, strength_1to5, etc.
    """
    G = nx.DiGraph()

    system_types = taxonomy.get("system_types", [])
    synonyms = taxonomy.get("synonyms", {})

    # --- Add nodes with standardized system_type
    for _, r in nodes_df.iterrows():
        node_id = str(r.get("node_id"))
        if not node_id or node_id == "nan":
            # Skip rows without usable node_id
            continue

        raw_sys = str(r.get("system_type", "") or "")
        sys_type = standardize_system_type(raw_sys, system_types, synonyms)

        # Copy all columns as attributes
        attrs = r.to_dict()
        attrs["system_type"] = sys_type  # overwrite with standardized value

        # Ensure numeric lat/lon if present
        for k in ("latitude", "longitude"):
            if k in attrs:
                try:
                    attrs[k] = float(attrs[k])
                except (TypeError, ValueError):
                    attrs[k] = None

        G.add_node(node_id, **attrs)

    # --- Add edges (only if both endpoints exist)
    for _, r in edges_df.iterrows():
        src = str(r.get("from_id"))
        dst = str(r.get("to_id"))
        if not src or not dst or src == "nan" or dst == "nan":
            continue
        if src not in G.nodes or dst not in G.nodes:
            # Keep graph clean; ignore dangling edges
            continue

        # Copy all edge columns as attributes (including from_id/to_id for traceability)
        G.add_edge(src, dst, **r.to_dict())

    return G


def compute_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute a basic set of network metrics per node:
        - degree (total)
        - in_degree
        - out_degree
        - betweenness (approx if needed)
    Returns a DataFrame keyed by node_id.
    """
    degree = dict(G.degree())
    indeg = dict(G.in_degree())
    outdeg = dict(G.out_degree())

    try:
        betw = nx.betweenness_centrality(G)
    except Exception:
        # Fallback if centrality fails for any reason
        betw = {n: 0.0 for n in G.nodes()}

    rows = []
    for n in G.nodes():
        rows.append(
            {
                "node_id": n,
                "degree": degree.get(n, 0),
                "in_degree": indeg.get(n, 0),
                "out_degree": outdeg.get(n, 0),
                "betweenness": betw.get(n, 0.0),
            }
        )

    return pd.DataFrame(rows)
