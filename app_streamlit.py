# app_streamlit.py
# Streamlit app for the OVW Gap Mapper:
# - Upload CSV/JSON templates (or use built-in defaults)
# - Toggle by system type (multi-lens)
# - Simulate vignettes to reveal missing links
# - Compute rural access gaps via distance-to-nearest service
#
# Run locally:
#   pip install streamlit pandas networkx plotly openpyxl
#   streamlit run app_streamlit.py

from pathlib import Path
import json
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# Add this near your other imports
try:
    from jsonschema import Draft7Validator
    _HAS_JSONSCHEMA = True
except Exception:
    _HAS_JSONSCHEMA = False

from build_graph import build_graph, load_taxonomy, load_settings, compute_metrics
from utils.geo import haversine_miles
from utils.vignettes import simulate_vignette

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="OVW Gap Mapper", layout="wide")
st.title("OVW Gap Mapper — Systems, Vignettes, and Gaps")
st.caption("Toggle by system type, simulate vignettes, and surface access gaps.")

# ---------------------------
# Optional: simple passcode gate using Streamlit Secrets
# Add APP_PASSCODE = "YourCode" in Streamlit Cloud → Settings → Secrets
# ---------------------------
PASS = st.secrets.get("APP_PASSCODE", None)

def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

if PASS:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
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
# Sidebar: Data Inputs
# ---------------------------
with st.sidebar:
    st.header("Load Data")
    nodes_file = st.file_uploader("Nodes CSV", type=["csv"])
    edges_file = st.file_uploader("Edges CSV", type=["csv"])
    taxonomy_file = st.file_uploader("Taxonomy JSON (optional)", type=["json"])
    settings_file = st.file_uploader("Settings JSON (optional)", type=["json"])
    vignettes_file = st.file_uploader("Vignettes JSON (optional)", type=["json"])

    st.markdown("---")
    st.caption("Tip: If you don't upload files, built-in templates will be used.")

# Fallbacks to bundled templates
default_dir = Path("data/templates")
def _read_default_csv(name):
    p = default_dir / name
    if p.exists():
        return pd.read_csv(p)
    st.error(f"Missing default file: {p}")
    st.stop()

def _read_default_json(name):
    p = default_dir / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    st.error(f"Missing default file: {p}")
    st.stop()

# Load data (either uploaded or defaults)
nodes_df = pd.read_csv(nodes_file) if nodes_file else _read_default_csv("nodes_template.csv")
edges_df = pd.read_csv(edges_file) if edges_file else _read_default_csv("edges_template.csv")
taxonomy = json.load(taxonomy_file) if taxonomy_file else _read_default_json("taxonomy_template.json")
settings = json.load(settings_file) if settings_file else _read_default_json("settings_template.json")
# --- Vignettes: load + (optional) schema validation ---
# If user uploads a file, we use it; otherwise we fall back to the template
if vignettes_file:
    vignettes_payload = json.load(vignettes_file)
else:
    vignettes_payload = json.loads((default_dir / "vignettes_template.json").read_text(encoding="utf-8"))

# Try to validate with JSON Schema if available
schema_path = Path("schemas/vignette.schema.json")
if schema_path.exists():
    if not _HAS_JSONSCHEMA:
        st.info(
            "Vignette schema found, but `jsonschema` is not installed. "
            "Add `jsonschema>=4.0` to requirements.txt to enable validation."
        )
    else:
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            validator = Draft7Validator(schema)
            errors = sorted(validator.iter_errors(vignettes_payload), key=lambda e: e.path)
            if errors:
                st.error("Uploaded vignettes JSON is invalid per schema. See details below.")
                # Render readable error paths (e.g., vignettes[0].title)
                for err in errors:
                    path = "root"
                    for p in err.path:
                        path += f"[{p}]" if isinstance(p, int) else f".{p}"
                    st.write(f"- **{path}**: {err.message}")
                st.stop()
        except Exception as e:
            st.warning(f"Schema validation skipped due to an error: {e}")
else:
    # No schema file present—continue without validation
    pass

# Basic validation hints
required_node_cols = {"node_id", "label", "system_type"}
required_edge_cols = {"from_id", "to_id"}

missing_nodes = required_node_cols.difference(nodes_df.columns)
missing_edges = required_edge_cols.difference(edges_df.columns)
if missing_nodes:
    st.warning(f"Nodes CSV is missing columns: {sorted(missing_nodes)}")
if missing_edges:
    st.warning(f"Edges CSV is missing columns: {sorted(missing_edges)}")

# ---------------------------
# Build Graph & Metrics
# ---------------------------
G = build_graph(nodes_df, edges_df, taxonomy)
metrics_df = compute_metrics(G)

# ---------------------------
# Lenses & Filters
# ---------------------------
system_types = taxonomy.get("system_types", [])
with st.sidebar:
    st.subheader("Lenses")
# Clean up whitespace and make sure system_type is a string
nodes_df["system_type"] = nodes_df["system_type"].astype(str).str.strip()

system_types = sorted(nodes_df["system_type"].dropna().unique().tolist())

selected_types = st.multiselect(
    "Show system types",
    options=system_types,
    default=system_types  # Default is “everything that exists” in the CSV
)
    lens = st.radio(
        "View",
        options=["By System Type", "By Referral Strength", "By Geography"],
        index=0
    )

# Apply system-type filter
if selected_types:
    keep_nodes = [n for n, d in G.nodes(data=True) if d.get("system_type") in selected_types]
    H = G.subgraph(keep_nodes).copy()
else:
    H = G.copy()

# Layout (spring)
pos = nx.spring_layout(H, seed=42, k=0.6)

# Colors by system type
type_order = list(dict.fromkeys([H.nodes[n].get("system_type", "Other") for n in H.nodes()]))
type_to_color_index = {t: i for i, t in enumerate(type_order)}

# Build Plotly traces
node_x, node_y, node_color, hover_text = [], [], [], []
for n in H.nodes():
    x, y = pos[n]
    node_x.append(x)
    node_y.append(y)
    d = H.nodes[n]
    hover_text.append(f"{d.get('label','')} ({n})<br>{d.get('system_type','')}")
    node_color.append(type_to_color_index.get(d.get("system_type", "Other"), 0))

edge_x, edge_y = [], []
for u, v in H.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y, mode="lines",
    line=dict(width=1), hoverinfo="none"
))
fig.add_trace(go.Scatter(
    x=node_x, y=node_y, mode="markers+text",
    textposition="top center",
    marker=dict(size=12, color=node_color, showscale=False),
    text=None,
    hovertext=hover_text, hoverinfo="text"
))
fig.update_layout(height=650, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))

st.subheader("Network View")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Metrics Table
# ---------------------------
st.subheader("Node Metrics")
metrics_join = metrics_df.merge(
    nodes_df[["node_id", "label", "system_type"]],
    how="left", on="node_id"
)
st.dataframe(metrics_join, use_container_width=True)

# ---------------------------
# Rural Access: Distance-to-Nearest Service
# ---------------------------
st.subheader("Rural Access — Distance-to-Nearest by Service Keyword")

default_threshold = settings.get("geography", {}).get("default_rural_threshold_miles", 30)
service_thresholds = settings.get("service_thresholds_miles", {}) or {"forensic_exam": 60}

service_key = st.selectbox(
    "Service keyword to check (searches in `services_offered` on nodes)",
    options=list(service_thresholds.keys()),
    index=0
)
threshold = service_thresholds.get(service_key, default_threshold)

def nearest_distance_miles(row: pd.Series, providers: pd.DataFrame):
    lat = row.get("latitude")
    lon = row.get("longitude")
    if pd.isna(lat) or pd.isna(lon):
        return None
    best = None
    for _, pr in providers.iterrows():
        plat, plon = pr.get("latitude"), pr.get("longitude")
        if pd.isna(plat) or pd.isna(plon):
            continue
        d = haversine_miles(float(lat), float(lon), float(plat), float(plon))
        best = d if (best is None or d < best) else best
    return best

providers = nodes_df[
    nodes_df["services_offered"].fillna("").str.contains(service_key, case=False, na=False)
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
# Vignettes — Journey Simulation
# ---------------------------
st.subheader("Vignettes — Journey Simulation")
vigs = vignettes_payload.get("vignettes", [])
if not vigs:
    st.info("No vignettes loaded. Provide a JSON or use the template to get started.")
else:
    titles = [f'{v["id"]}: {v.get("title","")}' for v in vigs]
    sel = st.selectbox("Choose vignette", options=titles, index=0)
    v = vigs[titles.index(sel)]
    result = simulate_vignette(H, v)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Journey Steps**")
        st.json(result["steps"])
    with c2:
        st.markdown("**Gaps Found**")
        st.json(result["gaps"])

# ---------------------------
# Downloads
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

st.caption("Tip: Edit taxonomy & thresholds in JSON templates. Upload CSVs or use the defaults in `data/templates`.")

