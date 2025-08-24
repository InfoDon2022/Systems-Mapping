"""
Systems Mapping App (Streamlit + PyVis)

Purpose: Web app to support OVW TA Initiative Purpose Area 18-style workshops
(SART-focused systems mapping for sexual assault response) with a simple, appealing UI.

How to run locally:
1) Install deps:  
   pip install streamlit pyvis pandas openpyxl networkx
2) Save this file as app.py
3) Run: streamlit run app.py

Excel template (optional upload):
- Sheet "Nodes" with columns (recommended):
  id,label,stage,shape,title,opacity,fontcolor,color,size,fixed,x,y
- Sheet "Edges" with columns: id,from,to,dashes
Only id,label are required on Nodes and id,from,to on Edges. Others are auto-filled.

Notes:
- This app avoids suicide-focused content. Vignettes align with DV/SA/stalking systems work.
- Editing: use the built-in tables and add-node / add-edge forms; the PyVis canvas is interactive
  for pan/zoom/select, but streamlit cannot capture in-canvas edit events reliably.
"""

import math
import io
import json
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from pyvis.network import Network
import networkx as nx

# ------------------------- UI CONFIG -------------------------
st.set_page_config(page_title="Systems Mapping – PA18", layout="wide")
st.markdown(
    """
    <style>
      .smallcaps {font-variant: small-caps; letter-spacing: 0.5px;}
      .legend span {display:inline-block; padding:4px 8px; margin-right:8px; border-radius:6px; font-size:12px;}
      .vignette {font-size:20px; line-height:1.5;}
      .stDownloadButton {margin-top: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- DEFAULT VIGNETTES (PA18-aligned) -------------------------
VIGNETTES: Dict[str, Dict[str, str]] = {
    "Reporting & Disclosure": {
        "Delayed Disclosure": "An adult survivor shares an assault that occurred months ago. She first told a trusted advocate at a community center and now is considering options, uncertain about evidence and timelines.",
        "Third‑Party Report": "A campus RA receives a disclosure from a student who does not want police involvement. The RA seeks guidance on confidential resources and mandated reporting boundaries.",
        "Military‑Connected Family": "A spouse of an active‑duty service member discloses sexual assault by another service member off‑base. She is unclear whether to pursue civilian, military, or both pathways.",
    },
    "Medical Forensic Care": {
        "SANE Availability": "A rural hospital with limited SANE coverage must coordinate transport to a regional site without causing undue burden or cost to the survivor.",
        "Pediatric Coordination": "A CAC (Children's Advocacy Center) seeks streamlined referrals for adolescent patients who present after an assault when the window for evidence collection is uncertain.",
        "Evidence Transfer": "Chain‑of‑custody protocols between hospital, law enforcement, and lab require clarity when survivors choose a non‑report kit first.",
    },
    "Law Enforcement & Prosecution": {
        "Trauma‑Informed Interview": "Investigators coordinate with advocates to plan survivor‑centered interviews and safety checks, minimizing duplication across agencies.",
        "Jurisdiction Maze": "Incident spans campus and city lines; clarifying primary jurisdiction, MOUs, and data sharing is urgent.",
        "Case Declination Loop": "A case is declined for filing; the SART examines feedback loops to improve documentation, evidence quality, and survivor communication.",
    },
    "Advocacy & Support Services": {
        "Warm Hand‑offs": "24/7 hotline, shelter, and counseling teams coordinate same‑day appointments and transportation without repeating intake questions.",
        "LEP Access": "A survivor with limited English proficiency requests services; teams activate language access (interpreters, translated forms) across partners.",
        "Disability Access": "Advocates coordinate accessible transportation and communication supports for a Deaf survivor engaging with multiple agencies.",
    },
    "Campus & Military Contexts": {
        "Clery/Title IX Alignment": "Campus Title IX works with local SART to align supportive measures, privacy, and parallel processes.",
        "Veterans Treatment Court": "A veteran defendant is in specialty court; the SART explores victim safety, no‑contact orders, and information flow respecting confidentiality.",
        "Base‑Civilian Bridge": "A local MOU clarifies referral routes between installation resources and community providers for dependents and retirees.",
    },
    "Housing & Safety Planning": {
        "Emergency Placement": "Advocates secure short‑term hotel vouchers while pursuing longer‑term housing protections under VAWA.",
        "Technology Safety": "Stalking concerns require safety planning around devices, accounts, and location tracking.",
        "Economic Stability": "Survivor needs coordinated childcare, job leave documentation, and access to compensation funds.",
    },
    "Rural & Access Barriers": {
        "Travel Burden": "Survivor faces a 90‑minute drive to the nearest SANE site; SART considers mobile response and teleSANE options.",
        "Confidentiality in Small Towns": "Overlap of roles increases risk of informal disclosures; partners formalize privacy and conflict‑of‑interest practices.",
        "Cultural Navigation": "Culturally‑specific services partner with SART to tailor outreach and safety plans respectfully.",
    },
}

# ------------------------- COLOR / SHAPE MAPS -------------------------
# Map a "stage" (formerly "intercept") to a consistent color used throughout workshops
STAGE_COLORS = {
    1: "#338423",  # e.g., Reporting/Intake
    2: "#B41E03",  # e.g., Medical/Forensic
    3: "#97560C",  # e.g., Law Enforcement/Prosecution
    4: "#386FBA",  # e.g., Advocacy/Support Services
    5: "#6B21A8",  # e.g., Courts/Campus/Military
}
DEFAULT_NODE_SIZE = 20

# ------------------------- STATE INIT -------------------------
if "nodes" not in st.session_state:
    st.session_state.nodes = pd.DataFrame(
        [
            {"id": 1, "label": "Hotline / Advocacy", "stage": 4},
            {"id": 2, "label": "Hospital / SANE", "stage": 2},
            {"id": 3, "label": "Law Enforcement", "stage": 3},
            {"id": 4, "label": "Prosecution", "stage": 3},
            {"id": 5, "label": "Title IX", "stage": 5},
        ]
    )
if "edges" not in st.session_state:
    st.session_state.edges = pd.DataFrame(
        [
            {"id": 1, "from": 1, "to": 2},
            {"id": 2, "from": 2, "to": 3},
            {"id": 3, "from": 3, "to": 4},
            {"id": 4, "from": 1, "to": 5},
        ]
    )

# ------------------------- HELPERS -------------------------

def coerce_nodes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Required
    if "id" not in df.columns or "label" not in df.columns:
        raise ValueError("Nodes sheet must include 'id' and 'label' columns.")
    # Optional w/ defaults
    if "stage" not in df.columns:
        df["stage"] = 1
    if "shape" not in df.columns:
        df["shape"] = "dot"
    if "title" not in df.columns:
        df["title"] = ""
    if "opacity" not in df.columns:
        df["opacity"] = 1.0
    if "fontcolor" not in df.columns:
        df["fontcolor"] = "black"
    if "color" not in df.columns:
        df["color"] = df["stage"].map(lambda s: STAGE_COLORS.get(int(s), "#6B21A8"))
    if "size" not in df.columns:
        df["size"] = DEFAULT_NODE_SIZE
    if "fixed" not in df.columns:
        df["fixed"] = False
    if "x" not in df.columns:
        df["x"] = pd.NA
    if "y" not in df.columns:
        df["y"] = pd.NA
    # Normalize types
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["stage"] = pd.to_numeric(df["stage"], errors="coerce").fillna(1).astype(int)
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(DEFAULT_NODE_SIZE)
    return df


def coerce_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "id" not in df.columns:
        # generate edge IDs if not present
        df["id"] = range(1, len(df) + 1)
    if not set(["from", "to"]).issubset(df.columns):
        raise ValueError("Edges sheet must include 'from' and 'to' columns.")
    if "dashes" not in df.columns:
        df["dashes"] = False
    # Normalize types
    for col in ["id", "from", "to"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def compute_pin_layout(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Compute total degree, pin top 5 nodes to nice coordinates; leave others free."""
    df_out = nodes.copy()
    # degree
    out_deg = edges.groupby("from").size().rename("outDeg")
    in_deg = edges.groupby("to").size().rename("inDeg")
    deg = pd.concat([out_deg, in_deg], axis=1).fillna(0)
    deg["totalDeg"] = deg.sum(axis=1)
    top = deg.sort_values("totalDeg", ascending=False).head(5).index.tolist()
    # set defaults
    if "fixed" not in df_out.columns:
        df_out["fixed"] = False
    coords = [
        (0, 300),
        (-300, 0),
        (300, 0),
        (-200, -300),
        (200, -300),
    ]
    for i, nid in enumerate(top):
        df_out.loc[df_out["id"] == nid, ["x", "y", "fixed"]] = [coords[i][0], coords[i][1], True]
    return df_out

def render_network(nodes: pd.DataFrame, edges: pd.DataFrame):
    from pyvis.network import Network
    import json

    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        notebook=False,
        directed=True,
    )

    # physics (same as before)
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=95,
        spring_strength=0.04,
        damping=0.30,
        overlap=0,
    )

    # ✅ pass PURE JSON (no 'const options =')
    vis_options = {
        "nodes": {
            "shadow": {"enabled": True, "size": 10},
            "font": {"face": "verdana", "size": 16, "background": "#E3DBAB"},
        },
        "edges": {"arrows": {"to": {"enabled": True}}, "smooth": False},
        "interaction": {"hover": True},
        "manipulation": {"enabled": False},
        "physics": {"enabled": True},
    }
    net.set_options(json.dumps(vis_options))

    # add nodes
    for _, r in nodes.iterrows():
        color = STAGE_COLORS.get(int(r["stage"]), "#6B21A8")
        net.add_node(
            int(r["id"]),
            label=str(r["label"]),
            shape=str(r.get("shape", "dot")),
            title=str(r.get("title", "")),
            color=color,
            opacity=float(r.get("opacity", 1.0)) if not pd.isna(r.get("opacity", 1.0)) else 1.0,
            fixed=bool(r.get("fixed", False)),
            x=None if pd.isna(r.get("x")) else int(r.get("x")),
            y=None if pd.isna(r.get("y")) else int(r.get("y")),
            size=float(r.get("size", DEFAULT_NODE_SIZE)),
        )

    # add edges
    for _, r in edges.iterrows():
        net.add_edge(int(r["from"]), int(r["to"]))

    return net

def excel_bytes(nodes: pd.DataFrame, edges: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        nodes.to_excel(writer, index=False, sheet_name="Nodes")
        edges.to_excel(writer, index=False, sheet_name="Edges")
    return buf.getvalue()

# ------------------------- SIDEBAR -------------------------
st.sidebar.markdown("<div class='smallcaps'><h2>Workshop Controls</h2></div>", unsafe_allow_html=True)

with st.sidebar.expander("Upload/Save Graph", expanded=True):
    up = st.file_uploader("Upload .xlsx (Nodes/Edges)", type=["xlsx"])
    if up is not None:
        try:
            ndf = pd.read_excel(up, sheet_name=0)
            edf = pd.read_excel(up, sheet_name=1)
            ndf = coerce_nodes(ndf)
            edf = coerce_edges(edf)
            ndf = compute_pin_layout(ndf, edf)
            st.session_state.nodes = ndf
            st.session_state.edges = edf
            st.success("Loaded and pinned top nodes by degree.")
        except Exception as e:
            st.error(f"Failed to load Excel: {e}")
    dl = excel_bytes(st.session_state.nodes, st.session_state.edges)
    st.download_button("Save current map (.xlsx)", data=dl, file_name="systems-map.xlsx")

with st.sidebar.expander("Vignette Panel", expanded=True):
    layout = st.radio("Show Vignette Panel", ["Collapsed", "Half Width"], index=0)
    cat = st.selectbox("Select Category", list(VIGNETTES.keys()))
    vignette = st.selectbox("Select Vignette", list(VIGNETTES[cat].keys()))
    st.caption("Use vignettes to focus discussion on hand‑offs, bottlenecks, and MOUs.")

with st.sidebar.expander("Add Node", expanded=False):
    new_label = st.text_input("Label", "New Partner/Step")
    new_stage = st.selectbox("Stage (color)", options=[1,2,3,4,5], index=0, help="Maps to color categories in legend")
    new_shape = st.selectbox("Shape", ["dot", "square", "triangle", "star", "diamond"], index=0)
    new_title = st.text_input("Tooltip (optional)", "")
    if st.button("➕ Add Node"):
        nodes = st.session_state.nodes
        new_id = (nodes["id"].max() if len(nodes) else 0) + 1
        row = {"id": int(new_id), "label": new_label.strip() or f"Node {new_id}", "stage": int(new_stage), "shape": new_shape, "title": new_title, "size": DEFAULT_NODE_SIZE}
        st.session_state.nodes = coerce_nodes(pd.concat([nodes, pd.DataFrame([row])], ignore_index=True))
        st.toast("Node added")

with st.sidebar.expander("Add Edge", expanded=False):
    node_ids = st.session_state.nodes["id"].astype(int).tolist()
    if node_ids:
        frm = st.selectbox("From", node_ids, key="edge_from")
        to = st.selectbox("To", node_ids, key="edge_to")
        if st.button("➕ Add Edge"):
            edges = st.session_state.edges
            new_eid = (edges["id"].max() if len(edges) else 0) + 1
            erow = {"id": int(new_eid), "from": int(frm), "to": int(to)}
            st.session_state.edges = coerce_edges(pd.concat([edges, pd.DataFrame([erow])], ignore_index=True))
            st.toast("Edge added")
    else:
        st.info("Add nodes first.")

# ------------------------- MAIN BODY -------------------------
st.markdown("# Systems Mapping for Coordinated Sexual Assault Response")
st.markdown("Designed for multi‑jurisdictional SART workshops (PA18). Pan/zoom the map, use tables to edit, and export your updates.")

# Legend
legend_html = "<div class='legend'>" + "".join(
    [f"<span style='background:{col}; color:white;'>{i}</span>" for i, col in STAGE_COLORS.items()]
) + "</div>"
st.markdown("#### Stage Legend" + legend_html, unsafe_allow_html=True)

# Vignette Panel Toggle
cols = st.columns([1, 1]) if layout == "Half Width" else [st.container()]

# Build network with pinned layout for top-degree nodes
nodes = compute_pin_layout(coerce_nodes(st.session_state.nodes), coerce_edges(st.session_state.edges))
edges = coerce_edges(st.session_state.edges)

# Map canvas
if layout == "Half Width":
    with cols[0]:
        st.subheader("Community Map")
        net = render_network(nodes, edges)
        net_html = net.generate_html(notebook=False)
        st.components.v1.html(net_html, height=820, scrolling=False)
    with cols[1]:
        st.subheader("Selected Vignette")
        st.markdown(f"<div class='vignette'>{VIGNETTES[cat][vignette]}</div>", unsafe_allow_html=True)
else:
    st.subheader("Community Map")
    net = render_network(nodes, edges)
    net_html = net.generate_html(notebook=False)
    st.components.v1.html(net_html, height=820, scrolling=False)

# Tabs for data editing
mtab, ntab, etab = st.tabs(["⚙️ Workshop Tips", "👥 Stakeholders (Nodes)", "🔗 Links (Edges)"])

with mtab:
    st.markdown(
        """
        **Facilitation tips**
        - Start with a vignette; ask the group to narrate the *first 3 hand‑offs* and where delays occur.
        - Color by stage to emphasize roles, not hierarchies. Add tooltips for local policies (kits, Title IX, MOUs).
        - Use the *Save* button often; export between sessions to preserve progress.
        - Document language access and disability access needs as node titles for accountability.
        """
    )

with ntab:
    st.info("Double‑click cells to edit. Use the sidebar forms to add new nodes or edges.")
    edited_nodes = st.data_editor(
        nodes,
        num_rows="dynamic",
        use_container_width=True,
        column_config={"stage": st.column_config.NumberColumn(help="1–5 color category")},
        key="nodes_editor",
    )
    if st.button("💾 Apply node edits"):
        try:
            st.session_state.nodes = coerce_nodes(edited_nodes)
            st.success("Nodes updated.")
        except Exception as e:
            st.error(f"Node update failed: {e}")

with etab:
    st.info("Create and edit links between partners. Direction implies hand‑off or information flow.")
    edited_edges = st.data_editor(
        edges,
        num_rows="dynamic",
        use_container_width=True,
        key="edges_editor",
    )
    if st.button("💾 Apply edge edits"):
        try:
            st.session_state.edges = coerce_edges(edited_edges)
            st.success("Edges updated.")
        except Exception as e:
            st.error(f"Edge update failed: {e}")

# Footer
st.caption("© Systems mapping prototype for OVW TA workshops (PA18). This tool does not collect identifying information and is intended for facilitated group use.")
