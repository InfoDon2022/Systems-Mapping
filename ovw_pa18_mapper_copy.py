#
# C://Users//harri//Documents//Systems-Mapping//ovw_pa18_mapper
#ovw_pa18_mapper.py
# Streamlit app: OVW Purpose Area 18 Systems Mapper (Domestic Violence)
# - Upload/download Excel (Nodes, Edges)
# - DV-specific vignettes and roles
# - Auto-pin top-5 central nodes
# - Gap checks (must-link paths), isolates, bottlenecks
# - PyVis (vis.js) interactive graph render
#
# Requires: streamlit, pandas, networkx, pyvis, openpyxl

import json
from io import BytesIO
from pathlib import Path
from PIL import Image

import pandas as pd
import streamlit as st
import networkx as nx
from pyvis.network import Network

ASSETS = Path(__file__).parent / "assets"
LOGO_WORD = ASSETS / "logo_transparent_background.png"     # full wordmark
LOGO_MARK = ASSETS / "logo_network_transparent.png"        # icon-only mark

# Use the icon-only mark as the browser tab icon
st.set_page_config(
    page_title="OVW Community Systems Mapping",
    page_icon=Image.open(LOGO_MARK),
    layout="wide",
)

st.sidebar.image(str(LOGO_WORD), use_container_width=50)
st.sidebar.markdown(
    """
    <hr style='margin-top:5px;margin-bottom:10px;border:1px solid #ccc;'/>
    <h3 style='text-align:center; color:#7030A0;'></h3>
    """,
    unsafe_allow_html=True
)

# --------------------------
# DV taxonomy & vignettes
# --------------------------
DV_ROLES = [
    "Survivor/Family",
    "DV/SA Advocacy",
    "24/7 Hotline",
    "Emergency Shelter",
    "Medical/ED",
    "Behavioral Health",
    "Law Enforcement",
    "Prosecution",
    "Public Defender",
    "Civil Court/Protection Orders",
    "Probation/Parole",
    "Child Welfare",
    "Schools/Youth",
    "Economic Supports/Workforce",
    "Housing/Homelessness System",
    "Immigration/Legal Aid",
    "Culturally Specific Org",
    "LGBTQ+ Org",
    "Faith/Community",
    "Tribal Gov/Family Advocacy Program",
    "Firearms Relinquishment Unit",
    "APIP/BIP (Abusive Partner Intervention)",
    "Tech Safety Program",
]

DV_INTERCEPTS = [
    "Help-Seeking",
    "Hotline",
    "Safety Planning",
    "Emergency Shelter",
    "Medical/ED",
    "Protection Order",
    "Criminal Charging",
    "Pretrial",
    "Sentencing",
    "Probation/Parole",
    "Longer-term Services",
    "Housing/Economic Stability",
    "Family/Child Services",
    "Tech Safety",
]

DV_VIGNETTES = {
    "Relationship & Safety": {
        "Safety planning after escalation":
            "Survivor experiences escalating coercive control (financial restrictions, surveillance). Needs quick safety plan including tech safety and emergency contacts.",
        "Children & visitation concerns":
            "Survivor has temporary custody; unsafe visitation exchanges at home. Needs safe exchange, supervised visitation, and coordinated court orders."
    },
    "Legal Processes": {
        "Protection order maze":
            "Survivor needs civil protection order; transportation barriers, language access needs, unclear about firearm surrender procedures.",
        "Parallel criminal & civil cases":
            "Criminal case filed; survivor seeks protective order and housing. Coordination needed among prosecutor, advocate, court, and housing provider."
    },
    "Housing & Economic": {
        "Emergency to stable housing":
            "Survivor with two children needs shelter tonight and a pathway to rapid rehousing and income supports within 60 days.",
        "Employment risk & benefits":
            "Survivor risks job loss due to court dates and relocation. Needs workplace safety plan and FMLA/benefit navigation."
    },
    "Health & Behavioral Health": {
        "ED disclosure":
            "Survivor discloses DV in the ED; staff need warm handoff to advocates, documentation for protection order, and safety planning.",
        "Trauma & substance use":
            "Survivor with PTSD and substance use needs non-punitive, integrated supports; avoid service exclusion for ‘noncompliance’."
    },
    "Technology-facilitated abuse": {
        "Stalking via devices":
            "Survivor suspects spyware/location sharing. Needs device audit, account resets, and legal guidance for evidence preservation.",
        "Harassment & doxxing":
            "Survivor faces online threats. Needs law enforcement referral, restraining order language, and advocate-led tech safety plan."
    },
    "Community & Culture": {
        "Culturally specific supports":
            "Survivor prefers culturally specific org; coordinate with mainstream services to avoid duplication and maintain trust.",
        "Rural access":
            "Long distances and limited providers; tele-advocacy, mobile advocacy, and transportation solutions are key."
    }
}

# Must-link expectations to flag gaps (edges that ideally exist)
MUST_LINKS = [
    ("24/7 Hotline", "Emergency Shelter"),
    ("Medical/ED", "DV/SA Advocacy"),
    ("Law Enforcement", "Firearms Relinquishment Unit"),
    ("Civil Court/Protection Orders", "Firearms Relinquishment Unit"),
    ("Prosecution", "DV/SA Advocacy"),
    ("Survivor/Family", "DV/SA Advocacy"),
    ("DV/SA Advocacy", "Housing/Homelessness System"),
    ("DV/SA Advocacy", "Immigration/Legal Aid"),
]

# --- Visual semantics
FUNCTION_SHAPE = {
    "access": "diamond",
    "authority": "square",
    "service": "dot",
    "risk": "triangle"
}

STATUS_COLOR = {
    "green": "#06D6A0",
    "yellow": "#FFD166",
    "red": "#EF476F",
    "gray": "#BDBDBD"
}

# --------------------------
# Helpers
# --------------------------
def default_nodes_edges():
    nodes = pd.DataFrame([
        # id, label, role, intercept, x, y, fixed, color, shape, size, attributes_json
        [1, "Survivor", "Survivor/Family", "Help-Seeking", None, None, False, "#FFD166", "dot", 22, "{}"],
        [2, "Hotline", "24/7 Hotline", "Hotline", None, None, False, "#EF476F", "diamond", 20, "{}"],
        [3, "Advocacy Center", "DV/SA Advocacy", "Safety Planning", None, None, False, "#118AB2", "dot", 20, "{}"],
        [4, "Emergency Shelter", "Emergency Shelter", "Emergency Shelter", None, None, False, "#06D6A0", "dot", 20, "{}"],
        [5, "Police", "Law Enforcement", "Criminal Charging", None, None, False, "#26547C", "square", 18, "{}"],
        [6, "Civil Court", "Civil Court/Protection Orders", "Protection Order", None, None, False, "#8338EC", "square", 18, "{}"],
        [7, "Prosecutor", "Prosecution", "Criminal Charging", None, None, False, "#8D99AE", "square", 18, "{}"],
        [8, "ED/Clinic", "Medical/ED", "Medical/ED", None, None, False, "#90BE6D", "dot", 18, "{}"],
        [9, "Housing Navigator", "Housing/Homelessness System", "Housing/Economic Stability", None, None, False, "#F9844A", "dot", 18, "{}"],
        [10, "Firearm Surrender Unit", "Firearms Relinquishment Unit", "Protection Order", None, None, False, "#D90429", "triangle", 18, "{}"],
    ], columns=["id","label","role","intercept","x","y","fixed","color","shape","size","attributes_json"])

    edges = pd.DataFrame([
        # id, from, to, label, weight, dashes
        [1, 1, 2, "Call/Chat/Text", 1.0, False],
        [2, 2, 3, "Warm handoff", 1.0, False],
        [3, 3, 4, "Shelter placement", 1.0, False],
        [4, 8, 3, "Bedside advocate", 1.0, True],
        [5, 6, 10, "Order → surrender", 1.0, False],
        [6, 5, 10, "Seizure referral", 1.0, True],
        [7, 3, 9, "Housing referral", 1.0, False],
        [8, 1, 6, "File CPO", 1.0, True],
        [9, 7, 3, "Victim services", 1.0, False],
    ], columns=["id","from","to","label","weight","dashes"])
    return nodes, edges

def ensure_schema(df_nodes, df_edges):
    # Fill missing required columns with defaults
    for col, default in [
        ("id", None), ("label", ""), ("role", ""), ("intercept", ""),
        ("function", ""), ("status", "gray"),
        ("x", None), ("y", None), ("fixed", False),
        ("color", "#cccccc"), ("shape", "dot"), ("size", 18), ("attributes_json", "{}")
    ]:
        if col not in df_nodes.columns:
            df_nodes[col] = default

    for col, default in [
        ("id", None), ("from", None), ("to", None),
        ("label", ""), ("link_status", "solid"),
        ("weight", 1.0), ("dashes", False)
    ]:
        if col not in df_edges.columns:
            df_edges[col] = default

        # Normalize types
        df_nodes["fixed"] = df_nodes["fixed"].astype(bool)
        df_nodes["size"] = pd.to_numeric(df_nodes["size"], errors="coerce").fillna(18).astype(int)
        df_edges["weight"] = pd.to_numeric(df_edges["weight"], errors="coerce").fillna(1.0)
        df_edges["dashes"] = df_edges["dashes"].astype(bool)
        return df_nodes, df_edges

def compute_centrality(df_nodes, df_edges):
    G = nx.DiGraph()
    for _, r in df_nodes.iterrows():
        G.add_node(int(r["id"]), role=r["role"])
    for _, r in df_edges.iterrows():
        if pd.notna(r["from"]) and pd.notna(r["to"]):
            G.add_edge(int(r["from"]), int(r["to"]))

    degree = nx.degree_centrality(G)
    betw = nx.betweenness_centrality(G, normalized=True)
    indeg = dict(G.in_degree())
    outdeg = dict(G.out_degree())

    cent = pd.DataFrame({
        "id": list(degree.keys()),
        "degree_centrality": list(degree.values()),
        "betweenness": [betw[k] for k in degree.keys()],
        "in_degree": [indeg[k] for k in degree.keys()],
        "out_degree": [outdeg[k] for k in degree.keys()],
    })
    return G, cent.sort_values("degree_centrality", ascending=False)

def auto_pin_top5(df_nodes, cent_df):
    top = cent_df.sort_values("degree_centrality", ascending=False)["id"].tolist()[:5]
    # Pin at fixed coordinates to anchor layout
    anchors = [
        (0, 300), (-300, 0), (300, 0), (-200, -300), (200, -300)
    ]
    for i, node_id in enumerate(top):
        df_nodes.loc[df_nodes["id"] == node_id, "x"] = anchors[i][0]
        df_nodes.loc[df_nodes["id"] == node_id, "y"] = anchors[i][1]
        df_nodes.loc[df_nodes["id"] == node_id, "fixed"] = True
    return df_nodes, top

def build_pyvis(df_nodes, df_edges, height="800px", physics=True):
    import json
    net = Network(height=height, width="100%", directed=True, bgcolor="#ffffff", font_color="#222")
    net.barnes_hut()

    # Add nodes (PA-18 semantics)
    for _, r in df_nodes.iterrows():
        nid = int(r["id"])

        # Semantics → visual
        func = str(r.get("function", "")).lower()
        stat = str(r.get("status", "gray")).lower()
        shape = FUNCTION_SHAPE.get(func, "dot")
        color = STATUS_COLOR.get(stat, "#BDBDBD")  # default gray

        # Hover: structured prompts
        title_bits = [
            f"<b>{r['label']}</b>",
            f"Role: {r.get('role','')}",
            f"Type: {func or 'n/a'} | Status: {stat or 'gray'}",
            f"Intercept: {r.get('intercept','')}"
        ]
        attrs_raw = r.get("attributes_json", "{}") or "{}"
        extra = {}
        try:
            extra = json.loads(attrs_raw)
        except Exception:
            try:
                extra = json.loads(attrs_raw.replace("'", '"'))
            except Exception:
                extra = {}
        label_map = {
            "what_happens": "What happens",
            "time_to_safety": "Time to safety",
            "next_step": "Next step"
        }
        for k, lbl in label_map.items():
            if k in extra and str(extra[k]).strip():
                title_bits.append(f"{lbl}: {extra[k]}")

        fixed = bool(r.get("fixed", False))
        x = r.get("x", None)
        y = r.get("y", None)
        kwargs = {}
        if fixed and pd.notna(x) and pd.notna(y):
            kwargs["fixed"] = True
            kwargs["x"] = int(x)
            kwargs["y"] = int(y)

        net.add_node(
            nid,
            label=str(r.get("label", nid)),
            title="<br>".join(title_bits),
            color=color,               # ← from status
            shape=shape,               # ← from function
            size=int(r.get("size", 18)),
            **kwargs
        )

    # Add edges (handoff strength semantics)
    for _, r in df_edges.iterrows():
        if pd.isna(r["from"]) or pd.isna(r["to"]):
            continue
        link_status = str(r.get("link_status", "solid")).lower()
        dashes = (link_status == "dashed") or bool(r.get("dashes", False))  # backward compatible
        width = float(r.get("weight", 1.0))

        # Optional: include a tiny “what happens / time to safety / next step” hover on edges too
        etitle = str(r.get("label", "") or "")
        net.add_edge(
            int(r["from"]), int(r["to"]),
            title=etitle,
            width=width,
            dashes=dashes,
            arrows="to"
        )

    # ✅ Set options with valid JSON (no 'const options =')
    options = {
        "physics": {
            "enabled": bool(physics),
            "solver": "barnesHut",
            "barnesHut": {"springLength": 95, "damping": 0.30}
        },
        "interaction": {"hover": True, "multiselect": True, "navigationButtons": True}
    }
    net.set_options(json.dumps(options))
    return net

def must_link_gaps(df_nodes, df_edges):
    # Map role -> ids
    role_to_ids = {}
    for _, r in df_nodes.iterrows():
        role_to_ids.setdefault(str(r["role"]), []).append(int(r["id"]))

    G = nx.DiGraph()
    G.add_nodes_from(df_nodes["id"].astype(int).tolist())
    for _, r in df_edges.iterrows():
        if pd.notna(r["from"]) and pd.notna(r["to"]):
            G.add_edge(int(r["from"]), int(r["to"]))

    gaps = []
    for a_role, b_role in MUST_LINKS:
        a_ids = role_to_ids.get(a_role, [])
        b_ids = role_to_ids.get(b_role, [])
        if not a_ids or not b_ids:
            gaps.append({"from_role": a_role, "to_role": b_role, "reason": "role_missing"})
            continue
        # If no path from any A to any B, flag
        reachable = False
        for a in a_ids:
            for b in b_ids:
                try:
                    if nx.has_path(G, a, b):
                        reachable = True
                        break
                except nx.NodeNotFound:
                    pass
            if reachable:
                break
        if not reachable:
            gaps.append({"from_role": a_role, "to_role": b_role, "reason": "no_path"})
    return pd.DataFrame(gaps)

def download_xlsx(df_nodes, df_edges):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_nodes.to_excel(writer, index=False, sheet_name="Nodes")
        df_edges.to_excel(writer, index=False, sheet_name="Edges")
    out.seek(0)
    return out

# --------------------------
# Session state init
# --------------------------
if "nodes" not in st.session_state or "edges" not in st.session_state:
    st.session_state["nodes"], st.session_state["edges"] = default_nodes_edges()

# Track anchor pinning so we can undo / unlock / re-lock
if "pin_state" not in st.session_state:
    st.session_state["pin_state"] = {
        "applied": False,
        "top_ids": [],
        "backup": None,   # DataFrame snapshot of nodes before pin
    }

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.markdown("## Workshop Controls")
uploaded = st.sidebar.file_uploader("Upload Map (.xlsx with Nodes & Edges)", type=["xlsx"])
if uploaded:
    try:
        nodes = pd.read_excel(uploaded, sheet_name="Nodes")
        edges = pd.read_excel(uploaded, sheet_name="Edges")
        nodes, edges = ensure_schema(nodes, edges)
        st.session_state["nodes"] = nodes
        st.session_state["edges"] = edges
        st.sidebar.success("Loaded!")
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")

with st.sidebar.expander("Add Node", expanded=False):
    with st.form("add_node"):
        new_id = int((pd.to_numeric(st.session_state["nodes"]["id"], errors="coerce").fillna(0).max() or 0) + 1)
        label = st.text_input("Label", value="")
        role = st.selectbox("Role", DV_ROLES, index=DV_ROLES.index("DV/SA Advocacy"))
        intercept = st.selectbox("Intercept (system stage)", DV_INTERCEPTS, index=DV_INTERCEPTS.index("Safety Planning"))
        color = st.color_picker("Color", "#118AB2")
        shape = st.selectbox("Shape", ["dot","square","diamond","triangle","triangleDown","star"], index=0)
        size = st.slider("Size", 10, 36, 20)
        fixed = st.checkbox("Pin position?", value=False)
        attrs = st.text_area("Extra attributes (JSON)", value="{}", height=80)
        submitted = st.form_submit_button("Add")
        if submitted:
            st.session_state["nodes"] = pd.concat([
                st.session_state["nodes"],
                pd.DataFrame([{
                    "id": new_id, "label": label, "role": role, "intercept": intercept,
                    "x": None, "y": None, "fixed": fixed, "color": color, "shape": shape,
                    "size": size, "attributes_json": attrs
                }])
            ], ignore_index=True)
            st.success(f"Added node {new_id}: {label}")

with st.sidebar.expander("Add Edge", expanded=False):
    with st.form("add_edge"):
        new_eid = int((pd.to_numeric(st.session_state["edges"]["id"], errors="coerce").fillna(0).max() or 0) + 1)
        from_id = st.number_input("From node id", value=1, step=1)
        to_id = st.number_input("To node id", value=2, step=1)
        elabel = st.text_input("Label", value="")
        weight = st.slider("Weight", 0.5, 4.0, 1.0, 0.1)
        dashes = st.checkbox("Dashed?", value=False)
        submitted_e = st.form_submit_button("Add")
        if submitted_e:
            st.session_state["edges"] = pd.concat([
                st.session_state["edges"],
                pd.DataFrame([{
                    "id": new_eid, "from": int(from_id), "to": int(to_id),
                    "label": elabel, "weight": weight, "dashes": dashes
                }])
            ], ignore_index=True)
            st.success(f"Added edge {new_eid}: {from_id} → {to_id}")

st.sidebar.markdown("---")
with st.sidebar.expander("Anchors (Top-5)", expanded=False):
    # Auto-pin
    if st.button("Auto-pin Top-5 Anchors"):
        # Snapshot for Undo
        st.session_state["pin_state"]["backup"] = st.session_state["nodes"].copy(deep=True)

        _, cent = compute_centrality(st.session_state["nodes"], st.session_state["edges"])
        nodes_pinned, top_ids = auto_pin_top5(st.session_state["nodes"].copy(), cent)
        st.session_state["nodes"] = nodes_pinned
        st.session_state["pin_state"]["applied"] = True
        st.session_state["pin_state"]["top_ids"] = top_ids
        st.success(f"Anchored nodes: {top_ids}")

    # Undo pin (full restore)
    if st.button("Undo Pin"):
        backup = st.session_state["pin_state"].get("backup")
        if backup is not None:
            st.session_state["nodes"] = backup
            st.session_state["pin_state"] = {"applied": False, "top_ids": [], "backup": None}
            st.success("Restored node positions to pre-pin state.")
        else:
            st.info("Nothing to undo.")

    # Unlock the pinned anchors to allow drag (keep coordinates, set fixed=False)
    if st.button("Unlock Anchors (keep positions)"):
        top_ids = st.session_state["pin_state"].get("top_ids", [])
        if top_ids:
            nodes = st.session_state["nodes"].copy()
            nodes.loc[nodes["id"].isin(top_ids), "fixed"] = False
            st.session_state["nodes"] = nodes
            st.success("Anchors unlocked; you can drag them now.")
        else:
            st.info("No anchors recorded. Run Auto-pin first.")

    # Re-lock currently tracked anchors (set fixed=True for those IDs)
    if st.button("Re-lock Anchors"):
        top_ids = st.session_state["pin_state"].get("top_ids", [])
        if top_ids:
            nodes = st.session_state["nodes"].copy()
            nodes.loc[nodes["id"].isin(top_ids), "fixed"] = True
            st.session_state["nodes"] = nodes
            st.success("Anchors re-locked.")
        else:
            st.info("No anchors recorded. Run Auto-pin first.")

    # Clear fixed on all nodes (nuclear option)
    if st.button("Clear 'fixed' on all nodes"):
        nodes = st.session_state["nodes"].copy()
        nodes["fixed"] = False
        st.session_state["nodes"] = nodes
        st.warning("All nodes are now draggable.")

xlsx_bytes = download_xlsx(st.session_state["nodes"], st.session_state["edges"])
st.sidebar.download_button("Save Map (.xlsx)", data=xlsx_bytes, file_name="DV_System_Map.xlsx")

# --------------------------
# Main tabs
# --------------------------
tab_map, tab_data, tab_vigs, tab_insights, tab_template = st.tabs(
    ["Map", "Data", "Vignettes", "Insights", "Templates"]
)

with tab_map:
    st.markdown("### Community Map")
    with st.expander("Legend", expanded=False):
        st.markdown("**Shapes:** ◆ Access | ■ Authority | ● Service | ▲ Risk")
        st.markdown("**Colors:** Green=working | Yellow=limited | Red=gap | Gray=unknown")
        st.markdown("**Edges:** Solid=standard handoff | Dashed=weak/provisional")
        st.caption("Tip: Use attributes_json per node like "
                   '`{"what_happens":"Warm handoff", "time_to_safety":"<2h", "next_step":"Finalize MOU"}`')
    physics = st.toggle("Enable physics layout", value=True, help="Turn off to lock current positions")
    net = build_pyvis(st.session_state["nodes"], st.session_state["edges"], physics=physics)
    html = net.generate_html(notebook=False)  # avoid notebook mode
    st.components.v1.html(html, height=820, scrolling=True)

with tab_data:
    st.markdown("#### Nodes")
    st.dataframe(st.session_state["nodes"], use_container_width=True)
    st.markdown("#### Edges")
    st.dataframe(st.session_state["edges"], use_container_width=True)

with tab_vigs:
    st.markdown("### Vignettes (for facilitated discussion)")
    cat = st.selectbox("Category", list(DV_VIGNETTES.keys()))
    item = st.selectbox("Scenario", list(DV_VIGNETTES[cat].keys()))
    st.info(DV_VIGNETTES[cat][item])
    st.caption("Prompt: Which roles should connect, at what stage, by which warm-handoffs? What breaks first when this scenario occurs at 2am on a weekend?")

with tab_insights:
    st.markdown("### Network Insights")
    G, cent = compute_centrality(st.session_state["nodes"], st.session_state["edges"])
    st.subheader("Top by degree centrality")
    st.dataframe(cent.head(10), use_container_width=True)

    # Isolates (no in/out edges)
    iso = [n for n in G.nodes() if G.in_degree(n)==0 and G.out_degree(n)==0]
    if iso:
        iso_df = st.session_state["nodes"][st.session_state["nodes"]["id"].isin(iso)][["id","label","role"]]
        st.warning("Isolated nodes (not connected):")
        st.dataframe(iso_df, use_container_width=True)
    else:
        st.success("No isolated nodes.")

    # Must-link gaps
    gaps_df = must_link_gaps(st.session_state["nodes"], st.session_state["edges"])
    if len(gaps_df):
        st.error("Must-link gaps (either role missing or no path):")
        st.dataframe(gaps_df, use_container_width=True)
        st.caption("Consider adding warm-handoffs or MOUs to cover these pathways.")
    else:
        st.success("All must-link pathways present.")

with tab_template:
    st.markdown("### Role & Intercept Reference")
    st.write("**Roles**:", ", ".join(DV_ROLES))
    st.write("**Intercepts**:", ", ".join(DV_INTERCEPTS))
    st.caption("Tip: keep node labels human-friendly (what people call them locally). Use role to classify.")

