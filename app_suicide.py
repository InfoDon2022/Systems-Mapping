"""
Vermont Suicide Prevention Systems Mapping App (Streamlit + PyVis)

This application is adapted from a sexual‑assault/OVW oriented systems mapping
tool to support the Vermont state suicide prevention initiative.  It is
intended for six‑hour, county‑level meetings where community partners
collaboratively build a network map of programs and hand‑offs in the local
suicide prevention ecosystem.  The interface allows participants to add and
edit nodes (partners/steps) and edges (relationships/hand‑offs), explore
scenario vignettes, and export their work.  A new performance dashboard
aligns with the Vermont Strategic Plan for Suicide Prevention (2024‑2029)
performance measures【923281966487039†L1236-L1252】.  Meeting facilitators can use
the dashboard to assess how the county is performing against statewide
targets, with simple green/yellow/red grading based on progress towards each
measure【923281966487039†L1350-L1367】.  Export buttons generate Excel files
containing the network and dashboard for further analysis.

To run locally:
1. Install dependencies:
   ``pip install streamlit pyvis pandas openpyxl networkx``
2. Save this file as ``app_suicide.py``
3. Run the app with ``streamlit run app_suicide.py``

Note: This tool does not collect identifying information and is intended
for facilitated group use only.
"""

from __future__ import annotations

import io
import math
import json
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st

# ------------------------- VIGNETTES -------------------------
# These scenario vignettes are designed to spark discussion about hand‑offs,
# gaps, and coordination in a suicide prevention context.  Each category
# reflects a major theme from the Vermont suicide prevention strategic plan.
VIGNETTES: Dict[str, Dict[str, str]] = {
    "Community Prevention & Awareness": {
        "Youth Outreach": (
            "A high-school counselor notices a student withdrawing from peers and missing classes. The counselor engages school mental health staff and reaches out to community youth programs to coordinate support."
        ),
        "Rural Isolation": (
            "A middle-aged farmer living in a remote area tells their primary care provider about loneliness and suicidal thoughts. Local volunteers, telehealth providers and a peer support group work together to offer connection and resources."
        ),
        "Media Messaging": (
            "After a news story about suicide uses stigmatizing language, community partners collaborate to train media outlets in responsible messaging and share guidelines for reporting."
        ),
    },
    "Treatment & Crisis Services": {
        "Emergency Department Referral": (
            "An individual presents at the emergency department following a suicide attempt. Hospital staff coordinate a warm hand-off to mobile crisis clinicians and arrange follow-up with a community mental health agency."
        ),
        "Crisis Line Coordination": (
            "The 988 Suicide & Crisis Lifeline receives a call from someone in acute crisis. The lifeline counselor dispatches a mobile crisis team and notifies a local peer support center for follow-up."
        ),
        "Inpatient to Outpatient Transition": (
            "A patient discharged from inpatient psychiatric care needs a coordinated step-down plan. The hospital, outpatient therapist and peer warm-line collaborate to ensure safety planning and continuity of care."
        ),
    },
    "Postvention & Recovery": {
        "School Postvention": (
            "Following the suicide of a student, the school district activates its postvention protocol. Crisis counselors, faith leaders and youth peer mentors offer support groups and debriefings to students, staff and families."
        ),
        "Workplace Postvention": (
            "An employee dies by suicide. The employer partners with community mental health providers to offer crisis debriefings, counseling and referrals for staff, while HR reviews policies related to leave and accommodations."
        ),
        "Loss Survivor Support": (
            "A bereaved family seeks help after losing a loved one to suicide. Local support groups, grief counselors and faith-based organizations coordinate wrap-around care and connection to resources."
        ),
    },
    "Data, Quality & Research": {
        "Data Reporting": (
            "County agencies compile suicide attempt and death data to submit to the statewide surveillance system. Partners discuss how to improve data completeness and timeliness."
        ),
        "Research Partnership": (
            "A university researcher studies risk factors for suicide among Vermont residents. Findings are presented to service providers and inform new program designs."
        ),
        "Quality Improvement": (
            "A crisis response program reviews follow-up call outcomes and identifies process improvements to increase engagement and satisfaction."
        ),
    },
    "Equity & Inclusion": {
        "LGBTQ+ Youth Safe Spaces": (
            "A youth center launches safe-space programming for LGBTQ+ adolescents. The center collaborates with schools, parents and mental health providers to ensure inclusive, affirming support."
        ),
        "Indigenous & BIPOC Communities": (
            "Tribal leaders and BIPOC community organizers adapt suicide prevention materials to be culturally resonant and co-facilitate outreach events."
        ),
        "Disability Access": (
            "Partners collaborate with the Vermont Center for the Deaf and Hard of Hearing to ensure crisis hotlines and services are accessible via text, video relay and other accommodations."
        ),
    },
}

# ------------------------- PERFORMANCE MEASURES -------------------------
# Each entry corresponds to a performance measure from Appendix 2 of the
# Vermont Strategic Plan for Suicide Prevention (2024–2029).  The targets
# specify the statewide goals and are used to compute progress in the
# dashboard.  Citations are included to reference the source text.
PERFORMANCE_MEASURES: List[Dict[str, Any]] = [
    {
        "id": "PM1",
        "description": "# of healthcare agencies implementing Zero Suicide",
        "target": 30,
        "citation": "【923281966487039†L1247-L1252】",
    },
    {
        "id": "PM2",
        "description": "# of trainings offered to mental health providers",
        "target": 48,
        "citation": "【923281966487039†L1253-L1258】",
    },
    {
        "id": "PM3",
        "description": "# of SDoH Community of Practices completed",
        "target": 1,
        "citation": "【923281966487039†L1270-L1274】",
    },
    {
        "id": "PM4",
        "description": "# of spaces for connection for people at disproportionate risk",
        "target": 3,
        "citation": "【923281966487039†L1278-L1283】",
    },
    {
        "id": "PM5",
        "description": "# of Centers of Excellence",
        "target": 1,
        "citation": "【923281966487039†L1286-L1295】",
    },
    {
        "id": "PM6",
        "description": "# of funding opportunities pursued annually",
        "target": 2,
        "citation": "【923281966487039†L1296-L1302】",
    },
    {
        "id": "PM7",
        "description": "# of AHS policy workgroups",
        "target": 1,
        "citation": "【923281966487039†L1303-L1307】",
    },
    {
        "id": "PM8",
        "description": "# of workforce members trained in culturally responsive care",
        "target": 300,
        "citation": "【923281966487039†L1322-L1328】",
    },
    {
        "id": "PM9",
        "description": "# of comprehensive suicide data reports published annually",
        "target": 1,
        "citation": "【923281966487039†L1336-L1339】",
    },
    {
        "id": "PM10",
        "description": "# of prevention recommendations based on SDoH analyses",
        "target": 3,
        "citation": "【923281966487039†L1341-L1346】",
    },
    {
        "id": "PM11",
        "description": "# of postvention trainings offered",
        "target": 8,
        "citation": "【923281966487039†L1358-L1363】",
    },
    {
        "id": "PM12",
        "description": "# of facilities participating in caring contacts",
        "target": 6,
        "citation": "【923281966487039†L1363-L1367】",
    },
    {
        "id": "PM13",
        "description": "# of stakeholder advisory coalitions for disproportionately affected groups",
        "target": 1,
        "citation": "【923281966487039†L1375-L1384】",
    },
]

# ------------------------- COLOR / STAGE MAP -------------------------
# Map each stage to a color.  The stages roughly align with the strategic
# directions (Community Prevention, Treatment & Crisis, Postvention & Recovery,
# Data/Research, Equity & Inclusion).
STAGE_COLORS = {
    1: "#C62828",  # Red
    2: "#FDD835",  # Yellow
    3: "#2E7D32",  # Green
}
DEFAULT_NODE_SIZE = 20

# ------------------------- STATE INIT -------------------------
# Initialise session state for nodes and edges if not already present.  The
# default network includes core partners involved in suicide prevention.  These
# can be edited or deleted by the user.
if "nodes" not in st.session_state:
    st.session_state.nodes = pd.DataFrame(
        [
            {"id": 1, "label": "988 Lifeline", "stage": 2},
            {"id": 2, "label": "Community Mental Health Agency", "stage": 2},
            {"id": 3, "label": "Hospital Emergency Dept", "stage": 2},
            {"id": 4, "label": "Peer Support Group", "stage": 3},
            {"id": 5, "label": "School & Youth Programs", "stage": 1},
            {"id": 6, "label": "Primary Care Clinic", "stage": 2},
            {"id": 7, "label": "Public Health Data Team", "stage": 1},
            {"id": 8, "label": "LGBTQ+ Safe Space", "stage": 1},
        ]
    )
if "edges" not in st.session_state:
    st.session_state.edges = pd.DataFrame(
        [
            {"id": 1, "from": 1, "to": 2},  # Lifeline → Community mental health
            {"id": 2, "from": 2, "to": 3},  # Community mental health → Hospital
            {"id": 3, "from": 3, "to": 6},  # Hospital → Primary Care
            {"id": 4, "from": 1, "to": 4},  # Lifeline → Peer Support
            {"id": 5, "from": 5, "to": 2},  # School → Community mental health
            {"id": 6, "from": 7, "to": 2},  # Data team → Community mental health (feedback)
        ]
    )

# ------------------------- HELPER FUNCTIONS -------------------------
def coerce_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and normalise node attributes."""
    df = df.copy()
    # Required columns
    if "id" not in df.columns or "label" not in df.columns:
        raise ValueError("Nodes sheet must include 'id' and 'label' columns.")
    # Optional columns with defaults
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
        df["color"] = df["stage"].map(lambda s: STAGE_COLORS.get(int(s), "#6A1B9A"))
    if "size" not in df.columns:
        df["size"] = DEFAULT_NODE_SIZE
    if "fixed" not in df.columns:
        df["fixed"] = False
    if "x" not in df.columns:
        df["x"] = pd.NA
    if "y" not in df.columns:
        df["y"] = pd.NA
    # Normalise types
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["stage"] = pd.to_numeric(df["stage"], errors="coerce").fillna(1).astype(int)
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(DEFAULT_NODE_SIZE)
    return df


def coerce_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and normalise edge attributes."""
    df = df.copy()
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)
    if not {"from", "to"}.issubset(df.columns):
        raise ValueError("Edges sheet must include 'from' and 'to' columns.")
    if "dashes" not in df.columns:
        df["dashes"] = False
    for col in ["id", "from", "to"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def compute_pin_layout(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Pin the top five nodes by total degree to specific coordinates."""
    df_out = nodes.copy()
    # Compute degree
    out_deg = edges.groupby("from").size().rename("outDeg")
    in_deg = edges.groupby("to").size().rename("inDeg")
    deg = pd.concat([out_deg, in_deg], axis=1).fillna(0)
    deg["totalDeg"] = deg.sum(axis=1)
    top = deg.sort_values("totalDeg", ascending=False).head(5).index.tolist()
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
    """Render the PyVis network for display in Streamlit."""
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
    # Configure physics
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=95,
        spring_strength=0.04,
        damping=0.30,
        overlap=0,
    )
    # Visual options
    vis_options = {
        "nodes": {
            "shadow": {"enabled": True, "size": 10},
            "font": {"face": "verdana", "size": 16, "background": "#E3DBAB"},
        },
        "edges": {"arrows": {"to": {"enabled": True}}, "smooth": False},
        "interaction": {"hover": True},
        "manipulation": {"enabled": true},
        "physics": {"enabled": True},
    }
    net.set_options(json.dumps(vis_options))
    # Add nodes
    for _, r in nodes.iterrows():
        color = STAGE_COLORS.get(int(r["stage"]), "#6A1B9A")
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
        )
    # Add edges
    for _, r in edges.iterrows():
        net.add_edge(int(r["from"]), int(r["to"]))
    return net


def excel_bytes(nodes: pd.DataFrame, edges: pd.DataFrame, perf: pd.DataFrame | None = None) -> bytes:
    """Serialize nodes, edges and optionally performance metrics to an Excel workbook."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        nodes.to_excel(writer, index=False, sheet_name="Nodes")
        edges.to_excel(writer, index=False, sheet_name="Edges")
        if perf is not None:
            perf.to_excel(writer, index=False, sheet_name="Performance")
    return buf.getvalue()


def grade_measure(value: float, target: float) -> str:
    """Assign a grade (Green, Yellow, Red) based on progress towards the target."""
    if value is None or pd.isna(value):
        return "Red"
    try:
        v = float(value)
    except Exception:
        return "Red"
    if v >= target:
        return "Green"
    elif v >= 0.5 * target:
        return "Yellow"
    else:
        return "Red"


# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="Systems Mapping – VT Suicide Prevention", layout="wide")
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

# ------------------------- BRANDING (Strategraph) -------------------------
cA, cB = st.columns([4, 1])
with cB:
    try:
        st.image("assets/logo_network_transparent.png", use_container_width=True)
    except Exception:
        try:
            st.image("assets/logo_transparent_background.png", use_container_width=True)
        except Exception:
            pass
st.markdown("<div style='text-align:right; font-weight:600;'>Strategraph</div>", unsafe_allow_html=True)


# ------------------------- SIDEBAR CONTROLS -------------------------
st.sidebar.markdown("## Workshop Controls")

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
    # Save current map (without performance sheet)
    dl_graph = excel_bytes(st.session_state.nodes, st.session_state.edges)
    st.download_button(
        "Save current map (.xlsx)",
        data=dl_graph,
        file_name="suicide-systems-map.xlsx",
    )

with st.sidebar.expander("Vignette Panel", expanded=True):
    layout_choice = st.radio("Show Vignette Panel", ["Collapsed", "Half Width"], index=0)
    cat = st.selectbox("Select Category", list(VIGNETTES.keys()))
    vignette_key = st.selectbox("Select Vignette", list(VIGNETTES[cat].keys()))
    st.caption("Use vignettes to focus discussion on hand‑offs, bottlenecks, and partnerships.")

with st.sidebar.expander("Add Node", expanded=False):
    new_label = st.text_input("Label", "New Partner/Step")
    new_stage = st.selectbox(
        "Stage (color)",
        options=[1, 2, 3],
        index=0,
        help="Maps to color categories in legend",
    )
    new_shape = st.selectbox("Shape", ["dot", "square", "triangle", "star", "diamond"], index=0)
    new_title = st.text_input("Tooltip (optional)", "")
    if st.button("➕ Add Node"):
        nodes_df = st.session_state.nodes
        new_id = (nodes_df["id"].max() if len(nodes_df) else 0) + 1
        row = {
            "id": int(new_id),
            "label": new_label.strip() or f"Node {new_id}",
            "stage": int(new_stage),
            "shape": new_shape,
            "title": new_title,
            "size": DEFAULT_NODE_SIZE,
        }
        st.session_state.nodes = coerce_nodes(pd.concat([nodes_df, pd.DataFrame([row])], ignore_index=True))
        st.toast("Node added")


with st.sidebar.expander("Add Edge", expanded=False):
    nodes_df_for_edge = st.session_state.nodes.copy()
    if not nodes_df_for_edge.empty:
        id_to_label = {int(r["id"]): str(r["label"]) for _, r in nodes_df_for_edge.iterrows()}
        node_ids = list(id_to_label.keys())

        frm = st.selectbox("From", node_ids, key="edge_from",
                           format_func=lambda i: id_to_label.get(i, str(i)))
        to = st.selectbox("To", node_ids, key="edge_to",
                          format_func=lambda i: id_to_label.get(i, str(i)))

        if st.button("➕ Add Edge"):
            edges_df = st.session_state.edges
            new_eid = (edges_df["id"].max() if len(edges_df) else 0) + 1
            erow = {"id": int(new_eid), "from": int(frm), "to": int(to)}
            st.session_state.edges = coerce_edges(
                pd.concat([edges_df, pd.DataFrame([erow])], ignore_index=True)
            )
            st.toast(f"Edge added: {id_to_label.get(frm, frm)} → {id_to_label.get(to, to)}")
    else:
        st.info("Add nodes first.")

with st.sidebar.expander("Performance Dashboard", expanded=False):
 expanded=False):
    st.markdown("### County Performance Assessment")
    st.caption(
        "Enter the current count for each measure below.  The app will compare your county’s progress against the statewide targets from Vermont’s suicide prevention plan and assign a color‑coded grade."
    )
    # Initialize performance values in session state
    if "performance_values" not in st.session_state:
        st.session_state.performance_values = {pm["id"]: 0 for pm in PERFORMANCE_MEASURES}
    # Input fields
    perf_inputs = {}
    for pm in PERFORMANCE_MEASURES:
        val = st.number_input(
            f"{pm['description']} (Target {pm['target']})",
            min_value=0.0,
            value=float(st.session_state.performance_values.get(pm["id"], 0)),
            step=1.0,
            key=f"input_{pm['id']}"
        )
        perf_inputs[pm["id"]] = val
    # Store back into session state
    st.session_state.performance_values.update(perf_inputs)
    # Compute summary dataframe
    summary_rows = []
    for pm in PERFORMANCE_MEASURES:
        val = perf_inputs.get(pm["id"], 0)
        grade = grade_measure(val, pm["target"])
        summary_rows.append({
            "Measure ID": pm["id"],
            "Description": pm["description"],
            "Current Value": val,
            "Target": pm["target"],
            "Grade": grade,
        })
    perf_df = pd.DataFrame(summary_rows)
    # Display table
    st.dataframe(perf_df, use_container_width=True)
    # Export performance sheet along with current map
    perf_bytes = excel_bytes(st.session_state.nodes, st.session_state.edges, perf_df)
    st.download_button(
        "Download dashboard (.xlsx)",
        data=perf_bytes,
        file_name="suicide-systems-dashboard.xlsx",
    )

# ------------------------- MAIN LAYOUT -------------------------
st.markdown("# Vermont Suicide Prevention Systems Map")
st.markdown(
    "This tool helps counties visualise their local suicide prevention systems, explore scenario vignettes, and assess performance against the state’s strategic measures."
)

# Legend
legend_html = " " + "".join([
    f"<span style='background:{col}; color:white; padding:4px 8px; border-radius:4px;'>{i}</span>"
    for i, col in STAGE_COLORS.items()
]) + " "
st.markdown("#### Stage Legend " + legend_html, unsafe_allow_html=True)

# Vignette panel toggle
cols = st.columns([1, 1]) if layout_choice == "Half Width" else [st.container()]

# Build network and pin layout
nodes_df = compute_pin_layout(coerce_nodes(st.session_state.nodes), coerce_edges(st.session_state.edges))
edges_df = coerce_edges(st.session_state.edges)

# Map canvas and vignette panel
if layout_choice == "Half Width":
    with cols[0]:
        st.subheader("Community Map")
        net = render_network(nodes_df, edges_df)
        net_html = net.generate_html(notebook=False)
        st.components.v1.html(net_html, height=820, scrolling=False)
    with cols[1]:
        st.subheader("Selected Vignette")
        st.markdown(f"<div class='vignette'>{VIGNETTES[cat][vignette_key]}</div>", unsafe_allow_html=True)
else:
    st.subheader("Community Map")
    net = render_network(nodes_df, edges_df)
    net_html = net.generate_html(notebook=False)
    st.components.v1.html(net_html, height=820, scrolling=False)

# Tabs for data editing
tips_tab, nodes_tab, edges_tab = st.tabs(["⚙️ Workshop Tips", "👥 Stakeholders (Nodes)", "🔗 Links (Edges)"])

with tips_tab:
    st.markdown(
        """
        **Facilitation Tips**
        - Begin with a vignette: ask participants to narrate the first hand‑offs and identify where delays or drop‑offs occur.
        - Use colors to emphasise role categories rather than hierarchies.  Add tooltips for local protocols (e.g., Zero Suicide policies, cultural adaptations).
        - Save your map frequently and export between sessions to preserve progress.
        - Document language access, cultural considerations and disability accommodations as node titles for accountability.
        """
    )

with nodes_tab:
    st.info("Double‑click cells to edit.  Use the sidebar forms to add new nodes or edges.")
    edited_nodes_df = st.data_editor(
        nodes_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={"stage": st.column_config.NumberColumn(help="1–5 color category")},
        key="nodes_editor",
    )
    if st.button("💾 Apply node edits"):
        try:
            st.session_state.nodes = coerce_nodes(edited_nodes_df)
            st.success("Nodes updated.")
        except Exception as e:
            st.error(f"Node update failed: {e}")

with edges_tab:
    st.info("Create and edit links between partners.  Direction implies hand‑off or information flow.")
    edited_edges_df = st.data_editor(
        edges_df,
        num_rows="dynamic",
        use_container_width=True,
        key="edges_editor",
    )
    if st.button("💾 Apply edge edits"):
        try:
            st.session_state.edges = coerce_edges(edited_edges_df)
            st.success("Edges updated.")
        except Exception as e:
            st.error(f"Edge update failed: {e}")

# Footer
st.caption("© Strategraph LLC")