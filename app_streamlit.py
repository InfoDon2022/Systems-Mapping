# app_streamlit.py
# OVW Gap Mapper — streamlined single-file workflow
# - Upload ONE Excel workbook (.xlsx) with sheets: "Nodes", "Edges"
# - Optional vignette JSON upload (built-ins provided)
# - Taxonomy & settings baked in (no uploads required)
# - Filter by system type; simulate vignettes to surface gaps
# - Export current graph (Nodes+Edges) back to Excel from the sidebar
#
# Run locally:
#   pip install streamlit pandas networkx plotly openpyxl jsonschema
#   streamlit run app_streamlit.py

from __future__ import annotations

import os
import json
from io import BytesIO
from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# Optional JSON schema validation for vignettes
try:
    from jsonschema import Draft7Validator
    _HAS_JSONSCHEMA = True
except Exception:
    _HAS_JSONSCHEMA = False

# --- local utilities (provide graceful fallbacks if missing) ---
try:
    from utils.standardize import standardize_system_type
except Exception:
    def standardize_system_type(raw: str, canon: List[str], synonyms: Dict[str, List[str]]) -> str:
        """Fallback: simple case-insensitive exact match on canon and synonym keys."""
        if not isinstance(raw, str):
            return "Other"
        s = raw.strip()
        # direct hit
        for c in canon:
            if s.lower() == c.lower():
                return c
        # synonym key hit
        for key in synonyms.keys():
            if s.lower() == key.lower():
                return key
            for alt in synonyms[key]:
                if s.lower() == str(alt).lower():
                    return key
        return "Other"

try:
    from utils.geo import haversine_miles
except Exception:
    from math import radians, sin, cos, asin, sqrt
    def haversine_miles(lat1, lon1, lat2, lon2) -> float:
        """Fallback Haversine in miles."""
        R = 3958.7613  # Earth radius in miles
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return R * c

try:
    from build_graph import build_graph, compute_metrics
except Exception as e:
    st.error(f"Could not import build_graph/compute_metrics: {e}")
    st.stop()

try:
    from utils.vignettes import simulate_vignette
except Exception:
    def simulate_vignette(G: nx.DiGraph, vignette: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback: check each consecutive pair in journey; report missing edges."""
        steps = []
        gaps = []
        journey = vignette.get("journey", [])
        for step in journey:
            # Normalize step forms used below
            f = step.get("from_node_id_or_label") or step.get("from") or step.get("from_label")
            t = step.get("to_node_label_or_id") or step.get("to") or step.get("to_label")
            steps.append({"from": f, "to": t})
            # Try to resolve by label first; fallback to id
            def _resolve(x):
                # id?
                try:
                    if x is not None and str(x).isdigit():
                        nid = int(x)
                        if nid in G.nodes:
                            return nid
                except Exception:
                    pass
                # label
                for n, d in G.nodes(data=True):
                    if str(d.get("label", "")).strip().lower() == str(x).strip().lower():
                        return n
                return None
            u = _resolve(f)
            v = _resolve(t)
            if u is None or v is None or (u, v) not in G.edges:
                gaps.append(f"Missing or unresolved link: {f} → {t}")
        return {"steps": steps, "gaps": gaps}

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="OVW Gap Mapper", layout="wide")
st.title("OVW Gap Mapper — Single-File Workflow")
st.caption("Upload one Excel workbook (Nodes + Edges). Optional: vignettes JSON. Taxonomy & settings are built-in.")

# ---------------------------
# Optional: passcode gate (env var first, then Streamlit secrets)
# Set APP_PASSCODE in your shell or in .streamlit/secrets.toml
# ---------------------------
PASS = os.environ.get("APP_PASSCODE") or st.secrets.get("APP_PASSCODE", None)

def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

if PASS:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        with st.sidebar:
            st.info("Enter access code to continue.")
            code = st.text_input("Access code", type="password")
        if not code:
            st.stop()
        if code == PASS:
            st.session_state.authenticated = True
            _safe_rerun()
        else:
            st.error("Incorrect code.")
            st.stop()
    else:
        with st.sidebar:
            if st.button("🔒 Logout"):
                st.session_state.authenticated = False
                _safe_rerun()

# ---------------------------
# Baked-in defaults (taxonomy, settings, vignettes)
# ---------------------------
DEFAULT_TAXONOMY = {
    "system_types": [
        "Advocacy","Law Enforcement","Prosecution","Courts","Healthcare","Forensic/SANE",
        "Behavioral Health","Housing/Shelter","Campus","Community-Based Org","Child/Youth Services",
        "Corrections/Probation/Parole","Legal Aid/Civil","Tribal Government/Programs",
        "Faith-Based Org","Hotline/Navigation","Transportation","Interpreter/Language Access","Individual","Other"
    ],
    "synonyms": {
        "Law Enforcement": ["Police","Sheriff","PD","Constable","State Police"],
        "Forensic/SANE": ["SANE","SAFE","Forensic Nurse","Forensic Examiner","SART Hospital"],
        "Healthcare": ["Hospital","Clinic","ED","ER","Health Center"],
        "Advocacy": ["Rape Crisis","DV Program","Victim Services","Victim Advocate","Coalition"],
        "Child/Youth Services": ["CPS","Child Protective Services","Youth Program","CAC","Children's Advocacy Center"],
        "Faith-Based Org": ["Church","Synagogue","Mosque","Temple","Faith Partner"],
        "Hotline/Navigation": ["Hotline","2-1-1","Navigation","Navigator"],
        "Interpreter/Language Access": ["Interpreter","Language Services","LEP Support"]
    }
}

DEFAULT_SETTINGS = {
    "geography": {"default_rural_threshold_miles": 30},
    "service_thresholds_miles": {
        "forensic_exam": 60,
        "advocacy": 40,
        "counseling": 40,
        "shelter": 50,
        "legal": 60
    }
}

BUILTIN_VIGNETTES = {
    "version": "1.1",
    "vignettes": [
        {
          "id": "vig_003",
          "title": "Military spouse survivor discloses to chaplain",
          "persona": "Spouse of active-duty service member near base housing",
          "scenario": "Seeks help from base chaplain → MP → advocacy.",
          "system_type": "Faith-Based Org",
          "geography": "Town",
          "preferred_path": ["Chaplain","Military Police","Advocacy Org","Legal Aid/Civil"],
          "observed_path":  ["Chaplain","Military Police","Advocacy Org"]
        },
        {
          "id": "vig_004",
          "title": "Teen in foster care discloses at school",
          "persona": "16-year-old in foster placement",
          "scenario": "Teacher → CPS → CAC → advocacy → court.",
          "system_type": "Child/Youth Services",
          "geography": "Town",
          "preferred_path": ["Teacher","CPS","Child Advocacy Center","Advocacy Org","Court"],
          "observed_path":  ["Teacher","CPS","Court"]
        },
        {
          "id": "vig_005",
          "title": "Tribal community survivor seeks care and legal follow-up",
          "persona": "Adult survivor in Tribal community",
          "scenario": "Tribal advocate → transport → regional hospital → legal.",
          "system_type": "Tribal Government/Programs",
          "geography": "Rural",
          "preferred_path": ["Tribal Advocate","Regional Hospital","Court"],
          "observed_path":  ["Tribal Advocate","Regional Hospital"]
        },
        {
          "id": "vig_006",
          "title": "Non-English speaker in rural area needs SANE access",
          "persona": "Adult survivor, LEP, limited transportation",
          "scenario": "Hotline → interpreter → SANE → advocacy.",
          "system_type": "Hotline/Navigation",
          "geography": "Rural",
          "preferred_path": ["Hotline/Navigation","Interpreter Service","SANE Program","Advocacy Org"],
          "observed_path":  ["Hotline/Navigation","Interpreter Service","Regional Hospital"]
        }
    ]
}

# ---------------------------
# Helpers
# ---------------------------
def _excel_to_frames(file) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read uploaded workbook; return (nodes_df, edges_df). Requires sheets 'Nodes' and 'Edges'."""
    try:
        xls = pd.ExcelFile(file)
        nodes_df = pd.read_excel(xls, sheet_name="Nodes")
        edges_df = pd.read_excel(xls, sheet_name="Edges")
    except Exception as e:
        st.error("Workbook must contain sheets named 'Nodes' and 'Edges'.")
        st.stop()

    # normalize column names and validate required ones
    nodes_df.columns = [c.strip() for c in nodes_df.columns]
    edges_df.columns = [c.strip() for c in edges_df.columns]

    required_nodes_cols = {"node_id", "label", "system_type"}
    required_edges_cols = {"from_id", "to_id"}

    missing_n = required_nodes_cols - set(nodes_df.columns)
    missing_e = required_edges_cols - set(edges_df.columns)
    if missing_n:
        st.error(f"Nodes sheet is missing required columns: {sorted(missing_n)}")
        st.stop()
    if missing_e:
        st.error(f"Edges sheet is missing required columns: {sorted(missing_e)}")
        st.stop()

    nodes_df["system_type"] = nodes_df["system_type"].astype(str).str.strip()
    return nodes_df, edges_df

def _build_journey_from_paths(v: Dict[str, Any]) -> List[Dict[str, Any]]:
    """If 'journey' missing, derive from observed_path or preferred_path."""
    if "journey" in v and isinstance(v["journey"], list):
        return v["journey"]
    path = v.get("observed_path") or v.get("preferred_path") or []
    journey = []
    if not path:
        return journey
    journey.append({"step": 1, "from": "start", "to_node_label_or_id": path[0]})
    step = 2
    for a, b in zip(path, path[1:]):
        journey.append({"step": step, "from_node_id_or_label": a, "to_node_label_or_id": b})
        step += 1
    return journey

def make_workbook_bytes(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> BytesIO:
    """Create an Excel workbook (Nodes + Edges) in memory for download."""
    # Canonical column order (add missing optional columns to keep shape consistent)
    nodes_cols = ["node_id", "label", "system_type", "latitude", "longitude", "services_offered"]
    edges_cols = ["from_id", "to_id", "relationship"]

    nd = nodes_df.copy()
    ed = edges_df.copy()

    for col in nodes_cols:
        if col not in nd.columns:
            nd[col] = ""
    if "relationship" not in ed.columns:
        ed["relationship"] = ""

    # Reorder; append any extra columns to the right
    extra_nodes = [c for c in nd.columns if c not in nodes_cols]
    extra_edges = [c for c in ed.columns if c not in edges_cols]
    nd = nd[nodes_cols + extra_nodes]
    ed = ed[edges_cols + extra_edges]

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        nd.to_excel(xw, sheet_name="Nodes", index=False)
        ed.to_excel(xw, sheet_name="Edges", index=False)
        # (Optional) README sheet
        # readme = pd.DataFrame({"Tip": [
        #     "This workbook has everything the app needs.",
        #     "Keep sheet names: 'Nodes' and 'Edges'.",
        #     "Required: Nodes(node_id,label,system_type), Edges(from_id,to_id)"
        # ]})
        # readme.to_excel(xw, sheet_name="README", index=False)
    buf.seek(0)
    return buf

# ---------------------------
# Sidebar: one workbook + optional vignettes + export
# ---------------------------
with st.sidebar:
    st.header("Workshop Data")
    xlsx = st.file_uploader("Upload Excel workbook (.xlsx) with 'Nodes' & 'Edges' sheets", type=["xlsx"])

    st.markdown("---")
    st.markdown("### Optional Vignettes")
    vfile = st.file_uploader("Upload custom vignettes JSON (optional)", type=["json"])
    st.caption("If not provided, the app uses built-in vignettes.")

    st.markdown("---")
    debug = st.checkbox("Show raw tables", value=False)

if not xlsx:
    st.info("Upload the **Excel workbook** to begin. Use the provided template.")
    st.stop()

# Load nodes/edges from the one workbook
nodes_df, edges_df = _excel_to_frames(xlsx)

# Standardize/validate system types based on baked-in taxonomy
canon_types = DEFAULT_TAXONOMY["system_types"]
synonyms = DEFAULT_TAXONOMY.get("synonyms", {})
nodes_df["system_type"] = nodes_df["system_type"].apply(
    lambda raw: standardize_system_type(raw, canon_types, synonyms)
)

# Build the graph (pass defaults for taxonomy/settings)
try:
    G = build_graph(nodes_df=nodes_df, edges_df=edges_df,
                    taxonomy=DEFAULT_TAXONOMY, settings=DEFAULT_SETTINGS)
except TypeError:
    # Compatibility: older signature without 'settings'
    G = build_graph(nodes_df, edges_df, DEFAULT_TAXONOMY)

# Metrics
metrics_df = compute_metrics(G)

# ---------------------------
# System-type lens (safe default = everything present)
# ---------------------------
sys_types = sorted(nodes_df["system_type"].dropna().unique().tolist())
selected_types = st.multiselect("Filter by system type", options=sys_types, default=sys_types)

# Apply filter
if selected_types:
    keep_nodes = [n for n, d in G.nodes(data=True) if d.get("system_type") in selected_types]
    H = G.subgraph(keep_nodes).copy()
else:
    H = G.copy()

# ---------------------------
# Simple network plot (spring layout)
# ---------------------------
pos = nx.spring_layout(H, seed=42, k=0.6)

# Build Plotly traces
edge_x, edge_y = [], []
for u, v in H.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

node_x, node_y, hover_text = [], [], []
# Color buckets by system type (simple index palette)
type_order = list(dict.fromkeys([H.nodes[n].get("system_type", "Other") for n in H.nodes()]))
type_to_color_index = {t: i for i, t in enumerate(type_order)}
node_color_idx = []

for n in H.nodes():
    x, y = pos[n]
    node_x.append(x); node_y.append(y)
    d = H.nodes[n]
    hover_text.append(f"{d.get('label','')} (id {n})<br>{d.get('system_type','')}")
    node_color_idx.append(type_to_color_index.get(d.get("system_type", "Other"), 0))

fig = go.Figure()
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                         line=dict(width=1), hoverinfo="none"))
fig.add_trace(go.Scatter(
    x=node_x, y=node_y, mode="markers+text",
    text=[H.nodes[n].get("label","") for n in H.nodes()],
    textposition="top center",
    marker=dict(size=12, color=node_color_idx, showscale=False),
    hovertext=hover_text, hoverinfo="text"
))
fig.update_layout(height=650, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))

st.subheader("Network View")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Export current graph back to Excel (unfiltered by default)
# ---------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### Export")
    st.caption("Download the current Nodes & Edges (as loaded/standardized).")
    wb = make_workbook_bytes(nodes_df, edges_df)
    st.download_button(
        label="📥 Download workbook (.xlsx)",
        data=wb,
        file_name="ovw_gap_mapper_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# ---------------------------
# Node Metrics table
# ---------------------------
st.subheader("Node Metrics")
metrics_join = metrics_df.merge(
    nodes_df[["node_id", "label", "system_type"]],
    how="left", on="node_id"
)
st.dataframe(metrics_join, use_container_width=True)

# ---------------------------
# Rural Access — distance-to-nearest provider
# ---------------------------
st.subheader("Rural Access — Distance-to-Nearest by Service Keyword")

default_threshold = DEFAULT_SETTINGS.get("geography", {}).get("default_rural_threshold_miles", 30)
service_thresholds = DEFAULT_SETTINGS.get("service_thresholds_miles", {}) or {"forensic_exam": 60}

service_key = st.selectbox(
    "Service keyword to check (searches in `services_offered` on nodes)",
    options=list(service_thresholds.keys()), index=0
)
threshold = service_thresholds.get(service_key, default_threshold)

def nearest_distance_miles(row: pd.Series, providers: pd.DataFrame):
    lat = row.get("latitude"); lon = row.get("longitude")
    try:
        lat = float(lat) if pd.notna(lat) and lat != "" else None
        lon = float(lon) if pd.notna(lon) and lon != "" else None
    except Exception:
        lat, lon = None, None
    if lat is None or lon is None:
        return None
    best = None
    for _, pr in providers.iterrows():
        plat, plon = pr.get("latitude"), pr.get("longitude")
        try:
            plat = float(plat) if pd.notna(plat) and plat != "" else None
            plon = float(plon) if pd.notna(plon) and plon != "" else None
        except Exception:
            plat, plon = None, None
        if plat is None or plon is None:
            continue
        d = haversine_miles(lat, lon, plat, plon)
        best = d if (best is None or d < best) else best
    return best

providers = nodes_df[
    nodes_df.get("services_offered", "").fillna("").str.contains(service_key, case=False, na=False)
]

if providers.empty:
    st.warning("No providers found with that service keyword. Update your nodes or thresholds.")

gap_rows = []
for _, seeker in nodes_df.iterrows():
    nd = nearest_distance_miles(seeker, providers) if not providers.empty else None
    gap_rows.append({
        "node_id": seeker["node_id"],
        "label": seeker["label"],
        "nearest_distance_miles": nd,
        "threshold_miles": threshold,
        "gap_flag": (nd is None) or (nd > threshold)
    })
gap_df = pd.DataFrame(gap_rows)
st.dataframe(gap_df, use_container_width=True)

# ---------------------------
# Vignettes — Journey Simulation (optional upload; else built-ins)
# ---------------------------
st.subheader("Vignettes — Journey Simulation")

# Load + validate (if schema exists)
try:
    vignettes_payload = json.load(vfile) if vfile else BUILTIN_VIGNETTES
except Exception as e:
    st.warning(f"Vignettes JSON could not be read; using built-ins. Error: {e}")
    vignettes_payload = BUILTIN_VIGNETTES

schema_path = Path("schemas/vignette.schema.json")
if vfile and schema_path.exists() and _HAS_JSONSCHEMA:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = Draft7Validator(schema)
        errors = sorted(validator.iter_errors(vignettes_payload), key=lambda e: e.path)
        if errors:
            st.error("Uploaded vignettes JSON is invalid per schema. See details below.")
            for err in errors:
                path = "root"
                for p in err.path:
                    path += f"[{p}]" if isinstance(p, int) else f".{p}"
                st.write(f"- **{path}**: {err.message}")
            st.stop()
    except Exception as e:
        st.warning(f"Schema validation skipped due to an error: {e}")
elif vfile and schema_path.exists() and not _HAS_JSONSCHEMA:
    st.info("Schema file found but `jsonschema` not installed. Add jsonschema>=4.0 to requirements.txt for validation.")

vignettes = vignettes_payload.get("vignettes", [])
if not vignettes:
    st.info("No vignettes available.")
else:
    titles = [f"{v.get('id','?')} — {v.get('title','(untitled)')}" for v in vignettes]
    idx = st.selectbox("Choose vignette", options=list(range(len(vignettes))), format_func=lambda i: titles[i])
    v = dict(vignettes[idx])
    v["journey"] = _build_journey_from_paths(v)

    # Simulate against the current (filtered) graph H
    # Ensure directed-graph semantics: if build_graph returns undirected, convert copy for checking
    if not isinstance(H, nx.DiGraph):
        Hd = nx.DiGraph()
        Hd.add_nodes_from(H.nodes(data=True))
        Hd.add_edges_from(H.edges(data=True))
    else:
        Hd = H

    result = simulate_vignette(Hd, v)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Journey Steps**")
        st.json(result.get("steps", []))
    with c2:
        st.markdown("**Gaps Found**")
        gaps = result.get("gaps", [])
        if gaps:
            st.error(f"{len(gaps)} gap(s) found.")
            st.json(gaps)
        else:
            st.success("No gaps detected for this vignette.")

# ---------------------------
# CSV Downloads (gap report & node metrics)
# ---------------------------
st.subheader("Download Reports")
st.download_button(
    "Download Gap Report (CSV)",
    gap_df.to_csv(index=False).encode("utf-8"),
    file_name="gap_report.csv",
    mime="text/csv",
)
st.download_button(
    "Download Node Metrics (CSV)",
    metrics_df.to_csv(index=False).encode("utf-8"),
    file_name="node_metrics.csv",
    mime="text/csv",
)

st.caption("Tip: Facilitators only need the one Excel workbook. Vignettes are optional; taxonomy & settings are built-in.")
