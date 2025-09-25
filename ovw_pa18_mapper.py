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
from datetime import datetime
import base64

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

st.sidebar.image(str(LOGO_WORD), use_container_width=True)
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
    ("24/7 Hotline", "DV/SA Advocacy"),
    ("Hospital/ED Intake", "SANE/Forensic Examiner"),
    ("SANE/Forensic Examiner", "Crime Lab/Forensics"),
    ("Law Enforcement", "Prosecution"),
    ("DV/SA Advocacy", "Hospital/ED Intake"),          # advocate notification at exam
    ("DV/SA Advocacy", "Law Enforcement"),             # advocate present at interview (as allowed)
    ("Campus/Military", "Law Enforcement"),            # when applicable
    ("Child Advocacy", "Law Enforcement"),             # minors/teen cases → CAC/MDT
]

# --- Visual semantics
FUNCTION_SHAPE = {
    "access": "diamond",
    "authority": "square",
    "service": "circle",
    "risk": "triangle"
}

STATUS_COLOR = {
    "green": "#27A539",
    "yellow": "#FFC000",
    "red": "#FF0000",
    "gray": "#BDBDBD"
}

## SART actor presets
def sart_core_nodes(starting_id:int):
    """Return a list[dict] of SART core nodes using your schema."""
    new = []
    nid = starting_id
    def add(label, role, intercept, function, status="gray", size=18):
        nonlocal nid
        new.append({
            "id": nid, "label": label, "role": role, "intercept": intercept,
            "function": function, "status": status, "x": None, "y": None,
            "fixed": False, "size": size, "attributes_json": "{}"
        })
        nid += 1

    # Core SART
    add("Survivor/Family",              "Survivor/Family",                "Help-Seeking",     "access",  size=18)
    add("24/7 Hotline",                 "24/7 Hotline",                   "Hotline",          "access")
    add("Hospital/ED Intake",           "Medical/ED",                     "Medical/ED",       "access")
    add("SANE/Forensic Examiner",       "Medical/ED",                     "Medical/ED",       "service")
    add("DV/SA Advocacy Center",        "DV/SA Advocacy",                 "Safety Planning",  "service")
    add("Law Enforcement (Sex Crimes)", "Law Enforcement",                "Criminal Charging","authority")
    add("Prosecutor’s Office",          "Prosecution",                    "Criminal Charging","authority")
    add("Crime Lab",                    "Crime Lab/Forensics",            "Criminal Charging","service")
    add("EMS/Dispatch (911)",           "EMS/Dispatch",                   "Help-Seeking",     "access")

    return new, nid



# --------------------------
# Helpers
# --------------------------

def build_mustlink_status(df_nodes, gaps_df, must_links):
    """Return a DataFrame with each must-link and its status."""
    missing = {(g["from_role"], g["to_role"]) for _, g in gaps_df.iterrows()} if len(gaps_df) else set()
    rows = []
    for a, b in must_links:
        status = "Missing" if (a, b) in missing else "Present"
        rows.append({"from_role": a, "to_role": b, "status": status})
    return pd.DataFrame(rows)

def export_kit_xlsx(nodes, edges, cent, iso_ids, gaps_df, must_links,
                    decisions_text="",
                    readiness_status=None, readiness_score=None, readiness_missing=None):

    import pandas as pd
    from datetime import datetime

    # ---- Safety copies / guards ----
    nodes = nodes.copy()
    edges = edges.copy()
    cent  = (cent if (cent is not None and not cent.empty)
             else pd.DataFrame(columns=["id","in_degree","out_degree","degree_centrality","betweenness"]))
    gaps_df = (gaps_df if gaps_df is not None else pd.DataFrame(columns=["from_role","to_role","status"]))
    iso_ids = (iso_ids if iso_ids is not None else [])

    # Isolated table (label/role, not raw ids only)
    iso_table = nodes.loc[nodes["id"].isin(iso_ids), ["id","label","role"]].reset_index(drop=True)

    # Must-link status table (derive status from gaps_df if provided)
    ml_status = pd.DataFrame(must_links, columns=["from_role","to_role"])
    if not gaps_df.empty and {"from_role","to_role"}.issubset(gaps_df.columns):
        missing = set(zip(gaps_df["from_role"], gaps_df["to_role"]))
        ml_status["status"] = ml_status.apply(
            lambda r: "Missing" if (r["from_role"], r["to_role"]) in missing else "Present", axis=1
        )
    else:
        ml_status["status"] = "Present"

    # ---- Overview blocks (ALWAYS define ov1 before appending) ----
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    overview_blocks = []

    ov1 = pd.DataFrame([{
        "Generated": now,
        "Nodes": int(len(nodes)),
        "Edges": int(len(edges)),
        "Isolated nodes": int(len(iso_ids)),
        "Must-link gaps": int(len(gaps_df)) if gaps_df is not None else 0
    }])
    overview_blocks.append(("Summary", ov1))

    # Readiness block
    rm_rows = [{
        "Status": readiness_status or "(not scored)",
        "Score": readiness_score if readiness_score is not None else "",
        "Missing elements": (
            ", ".join([m for m, ok in readiness_missing.items() if not ok])
            if isinstance(readiness_missing, dict)
            else (", ".join(readiness_missing) if readiness_missing else "")
        )
    }]
    ov_readiness = pd.DataFrame(rm_rows)
    overview_blocks.append(("Readiness", ov_readiness))

    # Key connectors (by betweenness) – show labels/roles
    ov2 = cent.sort_values(["betweenness","degree_centrality"], ascending=False) \
              .head(10)[["id","in_degree","out_degree","degree_centrality","betweenness"]]
    if not ov2.empty:
        ov2 = ov2.merge(nodes[["id","label","role"]], on="id", how="left")
        ov2 = ov2[["label","role","in_degree","out_degree","degree_centrality","betweenness"]]
    else:
        ov2 = pd.DataFrame(columns=["label","role","in_degree","out_degree","degree_centrality","betweenness"])
    overview_blocks.append(("Key connectors (by betweenness)", ov2))

    overview_blocks.append(("Isolated nodes", iso_table))
    overview_blocks.append(("Must-link status", ml_status))

    # Write workbook
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        # Overview: write blocks one under another
        row = 1
        for title, df in overview_blocks:
            df = df if len(df) else pd.DataFrame([{"—":"(none)"}])
            df.to_excel(xw, sheet_name="Overview", startrow=row, index=False)
            ws = xw.sheets["Overview"]
            ws.cell(row=row, column=1).value = title
            row += len(df) + 3

        # Raw sheets
        nodes.to_excel(xw, sheet_name="Nodes", index=False)
        edges.to_excel(xw, sheet_name="Edges", index=False)

        # Decisions/notes
        pd.DataFrame([{"Decisions/Next steps": decisions_text or "(none)"}])\
          .to_excel(xw, sheet_name="Decisions", index=False)
    out.seek(0)
    return out

def export_kit_html(nodes, edges, cent, iso_ids, gaps_df, must_links,
                    brand_png_path=None, decisions_text="",
                    readiness_status=None, readiness_score=None, readiness_missing=None):
    """Return a self-contained HTML one-pager."""
    ml_status = build_mustlink_status(nodes, gaps_df, must_links)
    # top connectors (by betweenness) – include labels
    top = cent.sort_values(["betweenness","degree_centrality"], ascending=False)\
              .head(5)[["id","in_degree","out_degree","degree_centrality","betweenness"]]
    top = top.merge(nodes[["id","label","role"]], on="id", how="left")
    top = top[["label","role","in_degree","out_degree","degree_centrality","betweenness"]]
    
    # Readiness summary text
    rm_status = readiness_status or "(not scored)"
    rm_score  = f"{readiness_score:.1f}/10" if isinstance(readiness_score,(int,float)) else "(n/a)"
    missing_list = []
    if readiness_missing:
        missing_list = [m for m, ok in readiness_missing.items() if not ok] \
                       if isinstance(readiness_missing, dict) else list(readiness_missing)
    missing_html = "<ul>" + "".join(f"<li>{m}</li>" for m in missing_list) + "</ul>" if missing_list else "<em>(none)</em>"

    iso_tbl = nodes[nodes["id"].isin(iso_ids)][["id","label","role"]] if iso_ids else pd.DataFrame()

    brand_img_html = ""
    if brand_png_path and Path(brand_png_path).exists():
        b64 = base64.b64encode(Path(brand_png_path).read_bytes()).decode()
        brand_img_html = f"<img src='data:image/png;base64,{b64}' style='height:42px;margin-right:10px;'>"

    def df_to_html(df):
        if df is None or len(df) == 0:
            return "<em>(none)</em>"
        return df.to_html(index=False, classes='table', border=0)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""
<!doctype html>
<html><head><meta charset="utf-8">
<title>Community Systems Mapping - Export Kit</title>
<style>
 body {{ font-family: Arial, sans-serif; margin: 24px; color:#222; }}
 h1,h2 {{ margin: 0 0 8px 0; }}
 .muted {{ color:#666; font-size: 12px; }}
 .section {{ margin-top: 18px; }}
 .table td, .table th {{ padding:6px 8px; border-bottom:1px solid #eee; }}
 .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#f3f3f3; margin-left:6px; }}
</style>
</head><body>
<div style="display:flex;align-items:center;gap:10px;">
  {brand_img_html}
  <div>
    <h1>OVW Community Systems Mapping — Export Kit</h1>
    <div class="muted">Generated {now}</div>
  </div>
</div>

<div class="section">
  <h2>Summary</h2>
  <div>Nodes: <span class="pill">{len(nodes)}</span>
       Edges: <span class="pill">{len(edges)}</span>
       Isolated: <span class="pill">{len(iso_ids)}</span>
       Must-link gaps: <span class="pill">{len(gaps_df)}</span></div>
</div>
<div class="section">
  <h2>Readiness</h2>
  <div>Status: <span class="pill">{rm_status}</span> &nbsp; Score: <span class="pill">{rm_score}</span></div>
  <div class="muted" style="margin-top:6px;">Missing or incomplete elements:</div>
  {missing_html}
</div>

<div class="section">
  <h2>Key connectors (by betweenness)</h2>
  <div class="muted">Roles that many survivor pathways cross; strain here slows the whole system.</div>
  {df_to_html(top)}
</div>
<div class="section"><h2>Isolated Partners</h2>{df_to_html(iso_tbl)}</div>
<div class="section"><h2>Must-link status</h2>{df_to_html(ml_status)}</div>

<div class="section">
  <h2>Decisions & Next Steps</h2>
  <div>{(decisions_text or '(none)').replace('\n','<br>')}</div>
</div>

</body></html>
"""
    return html

def readiness_score(checks:dict, metrics:dict):
    """Return ('green'|'yellow'|'red', detail) from booleans & SLA-like metrics."""
    # Required docs/structures (weight x2)
    doc_keys = ["has_charter","has_protocols","has_mous","case_review_cadence"]
    doc_ok = sum(1 for k in doc_keys if checks.get(k, False))
    # Participation (all core roles present)
    part_ok = checks.get("all_core_roles_present", False)
    # Victim-centered SLAs (simple thresholds; adjust if you like)
    time_adv = metrics.get("hours_to_advocate", None)     # target <= 2
    time_exam = metrics.get("hours_to_exam", None)        # target <= 24
    time_lab  = metrics.get("hours_to_lab", None)         # target <= 48
    sla_hits = 0
    for v, target in [(time_adv,2),(time_exam,24),(time_lab,48)]:
        if v is not None and v <= target: sla_hits += 1

    # Score 0–10
    score = (doc_ok*2) + (2 if part_ok else 0) + sla_hits  # max 4*2 + 2 + 3 = 13 → scale
    # Normalize to 0–10
    score = min(10, round(score * (10/13), 1))
    if score >= 7.5: status="green"
    elif score >= 5.0: status="yellow"
    else: status="red"
    return status, {"doc_ok":doc_ok, "part_ok":part_ok, "sla_hits":sla_hits, "score":score}

def default_nodes_edges():
    nodes = pd.DataFrame([
        # id, label, role, intercept, x, y, fixed, color, shape, size, attributes_json
        [1, "Survivor", "Survivor/Family", "Help-Seeking", None, None, False, "#FFD166", "circle", 18, "{}"],
        [2, "Hotline", "24/7 Hotline", "Hotline", None, None, False, "#EF476F", "diamond", 18, "{}"],
        [3, "Advocacy Center", "DV/SA Advocacy", "Safety Planning", None, None, False, "#118AB2", "circle", 18, "{}"],
        [4, "Emergency Shelter", "Emergency Shelter", "Emergency Shelter", None, None, False, "#06D6A0", "circle", 18, "{}"],
        [5, "Police", "Law Enforcement", "Criminal Charging", None, None, False, "#26547C", "square", 18, "{}"],
        [6, "Civil Court", "Civil Court/Protection Orders", "Protection Order", None, None, False, "#8338EC", "square", 18, "{}"],
        [7, "Prosecutor", "Prosecution", "Criminal Charging", None, None, False, "#8D99AE", "square", 18, "{}"],
        [8, "ED/Clinic", "Medical/ED", "Medical/ED", None, None, False, "#90BE6D", "circle", 18, "{}"],
        [9, "Housing Navigator", "Housing/Homelessness System", "Housing/Economic Stability", None, None, False, "#F9844A", "circle", 18, "{}"],
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
# --- Nodes: add any missing columns with safe defaults ---
    for col, default in [
        ("id", None), ("label", ""), ("role", ""), ("intercept", ""),
        ("function", ""), ("status", "gray"),
        ("specialized_training", ""), ("needs_training_on", ""),
        ("has_mou", ""), ("has_protocols", ""),
        ("x", None), ("y", None), ("fixed", False),
        ("color", "#cccccc"), ("shape", "dot"), ("size", 18),
        ("attributes_json", "{}"),
    ]:
        if col not in df_nodes.columns:
            df_nodes[col] = default

    # Edges: add any missing columns with safe defaults
    for col, default in [
        ("id", None), ("from", None), ("to", None),
        ("label", ""), ("link_status", "solid"),
        ("weight", 1.0), ("dashes", False),
    ]:
        if col not in df_edges.columns:
            df_edges[col] = default

    # Normalize types (🔧 moved OUTSIDE the loop)
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

def build_pyvis(df_nodes, df_edges, height="800px", physics=True,
                size_mode="Uniform", degree_centrality_map=None):
    import json
    net = Network(height=height, width="100%", directed=True, bgcolor="#ffffff", font_color="#222")
    net.barnes_hut()

    # Map centrality (0..1) → visual size range
    def size_for(nid: int, default=18):
        # Uniform mode or no centrality map → return default
        if str(size_mode).lower().startswith("uniform") or not degree_centrality_map:
            return default
        # Scaled mode (by degree centrality)
        dc = float(degree_centrality_map.get(int(nid), 0.0))
        lo, hi = 14, 30  # keep subtle so nothing gets cartoonishly large
        dc = max(0.0, min(1.0, dc))
        return int(lo + (hi - lo) * dc)

    # Add nodes (PA-18 semantics)
    for _, r in df_nodes.iterrows():
        nid = int(r["id"])

        # Semantics → visual
        func = str(r.get("function", "")).lower()
        stat = str(r.get("status", "gray")).lower()
        shape = FUNCTION_SHAPE.get(func, "circle")
        color = STATUS_COLOR.get(stat, "#BDBDBD")  # default gray

        # Hover: structured prompts
        title_bits = [
            f"<b>{r['label']}</b>",
            f"Role: {r.get('role','')}",
            f"Type: {func or 'n/a'} | Status: {stat or 'gray'}",
            f"Intercept: {r.get('intercept','')}"
        ]
        train = str(r.get("specialized_training","")).strip()
        needs = str(r.get("needs_training_on","")).strip()
        if train:
            title_bits.append(f"Training: {train}")            # yes | partial | no
        if needs:
            title_bits.append(f"Needs training on: {needs}")   # comma-separated topics

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

        # Fixed size (Uniform) OR scaled by degree if requested & available
        FIXED_NODE_SIZE = 18
        size_val = FIXED_NODE_SIZE

        if str(size_mode).lower().startswith("scaled") and degree_centrality_map:
            # centrality 0..1 → map to a gentle visual range
            dc = float(degree_centrality_map.get(int(nid), 0.0))
            lo, hi = 18, 20   # adjust if you want more/less contrast
            dc = max(0.0, min(1.0, dc))
            size_val = int(lo + (hi - lo) * dc)

        net.add_node(
            nid,
            label=str(r.get("label", nid)),
            title="<br>".join(title_bits),
            color=color,
            shape=shape,
            size=size_val,
            **kwargs
        )

    # Add edges (handoff strength semantics)
    for _, r in df_edges.iterrows():
        if pd.isna(r["from"]) or pd.isna(r["to"]):
            continue
        link_status = str(r.get("link_status", "solid")).lower()
        dashes = (link_status == "dashed") or bool(r.get("dashes", False))  # backward compatible
        width = float(r.get("weight", 1.0))

        etitle = str(r.get("label", "") or "")
        net.add_edge(
            int(r["from"]), int(r["to"]),
            title=etitle,
            width=width,
            dashes=dashes,
            arrows="to"
        )

    # Options (valid JSON)
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
        shape = st.selectbox("Shape", ["circle","square","diamond","triangle","triangleDown","star"], index=0)
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

with st.sidebar.expander("Add SART Core Set", expanded=False):
    add_optional_campus   = st.checkbox("Include Campus/Military liaison", value=False)
    add_optional_cac      = st.checkbox("Include CAC/Child MDT (for minors)", value=False)
    if st.button("Add SART nodes"):
        nodes = st.session_state["nodes"].copy()
        next_id = int(pd.to_numeric(nodes["id"], errors="coerce").fillna(0).max() or 0) + 1
        core, next_id = sart_core_nodes(next_id)

        # Optional partners
        if add_optional_campus:
            core.append({
                "id": next_id, "label": "Campus/Military Liaison", "role": "Campus/Military",
                "intercept": "Help-Seeking", "function": "access", "status": "gray",
                "x": None, "y": None, "fixed": False, "size": 18, "attributes_json": "{}"
            }); next_id += 1
        if add_optional_cac:
            core.append({
                "id": next_id, "label": "CAC / Child MDT", "role": "Child Advocacy",
                "intercept": "Family/Child Services", "function": "service", "status": "gray",
                "x": None, "y": None, "fixed": False, "size": 18, "attributes_json": "{}"
            }); next_id += 1

        st.session_state["nodes"] = pd.concat([nodes, pd.DataFrame(core)], ignore_index=True)
        st.success(f"Added {len(core)} SART nodes.")

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

with st.sidebar.expander("Display: Node Size", expanded=False):
    size_mode = st.radio(
        "Sizing mode",
        options=["Uniform", "Scaled (by degree)"],
        index=0,
        help="Uniform = same size for all nodes. Scaled = larger nodes have more connections."
    )
st.session_state["size_mode"] = size_mode

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
    degree_map = None
    if st.session_state.get("size_mode","Uniform").startswith("Scaled"):
        # Reuse your existing compute_centrality function; it should return a DataFrame with columns 'id' and 'degree_centrality'
        _, cent = compute_centrality(st.session_state["nodes"], st.session_state["edges"])
        degree_map = dict(zip(cent["id"].astype(int), cent["degree_centrality"].astype(float)))

    net = build_pyvis(
        st.session_state["nodes"],
        st.session_state["edges"],
        physics=physics,
        size_mode=st.session_state.get("size_mode","Uniform"),
        degree_centrality_map=degree_map
    )

    html = net.generate_html(notebook=False)
    st.components.v1.html(html, height=820, scrolling=True)

with tab_data:
    st.markdown("### Data (Editable)")

    st.caption("Tip: Double-click a cell to edit. Use the + button at the bottom to add rows. Changes update the map automatically.")

    # --- Editable Nodes table ---
    edited_nodes = st.data_editor(
        st.session_state["nodes"],
        num_rows="dynamic",                     # allow add/remove rows
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("id", step=1, help="Unique integer ID"),
            "label": st.column_config.TextColumn("label"),
            "role": st.column_config.TextColumn("role"),
            "intercept": st.column_config.TextColumn("intercept"),
            # Your simplified visuals:
            "function": st.column_config.SelectboxColumn(
                "function", options=["access","authority","service","risk"],
                help="Determines shape (diamond, square, circle, triangle)"
            ),
            "status": st.column_config.SelectboxColumn(
                "status", options=["green","yellow","red","gray"],
                help="Determines node color"
            ),
            "fixed": st.column_config.CheckboxColumn("fixed"),
            "x": st.column_config.NumberColumn("x", step=10, help="Optional fixed X"),
            "y": st.column_config.NumberColumn("y", step=10, help="Optional fixed Y"),
            "size": st.column_config.NumberColumn("size", min_value=10, max_value=48, step=1),
            "attributes_json": st.column_config.TextColumn(
                "attributes_json",
                help='Optional hover fields. Use valid JSON (e.g., {"what_happens":"...", "time_to_safety":"...", "next_step":"..."})',
                width="large"
            ),
            # Quiet fields used by Insights/Readiness (won’t affect map visuals)
            "specialized_training": st.column_config.SelectboxColumn(
                "specialized_training", options=["","yes","partial","no"], help="For training gaps table"
            ),
            "needs_training_on": st.column_config.TextColumn(
                "needs_training_on", help="Comma-separated topics (e.g., trauma, adolescent exams)"
            ),
            "has_mou": st.column_config.SelectboxColumn(
                "has_mou", options=["","yes","no"], help="Used by readiness checks"
            ),
            "has_protocols": st.column_config.SelectboxColumn(
                "has_protocols", options=["","yes","no"], help="Used by readiness checks"
            ),
        }
    )

    st.markdown("### Edges (Editable)")
    edited_edges = st.data_editor(
        st.session_state["edges"],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("id", step=1),
            "from": st.column_config.NumberColumn("from", step=1, help="Source node id"),
            "to": st.column_config.NumberColumn("to", step=1, help="Target node id"),
            "label": st.column_config.TextColumn("label", help="What this handoff is"),
            "link_status": st.column_config.SelectboxColumn(
                "link_status", options=["solid","dashed"], help="Solid=working, Dashed=aspirational/weak"
            ),
            "dashes": st.column_config.CheckboxColumn("dashes", help="Backward compatibility; will be set from link_status"),
            "weight": st.column_config.NumberColumn("weight", min_value=0.5, max_value=5.0, step=0.1),
        }
    )

    # --- Persist edits back into session state and normalize schema (safe defaults) ---
    # (ensure_schema should already add missing columns like specialized_training, etc.)
    edited_nodes, edited_edges = ensure_schema(edited_nodes.copy(), edited_edges.copy())

    # Auto-derive 'dashes' from 'link_status' for backward compatibility
    if "link_status" in edited_edges.columns:
        edited_edges["dashes"] = edited_edges["link_status"].astype(str).str.lower().eq("dashed")

    st.session_state["nodes"] = edited_nodes
    st.session_state["edges"] = edited_edges

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
    
    st.subheader("Training gaps (self-reported)")
    nodes = st.session_state["nodes"]
    cols = ["label","role","specialized_training","needs_training_on"]
    train_df = nodes.reindex(columns=cols).copy()
    train_df["specialized_training"] = train_df["specialized_training"].fillna("")
    train_df["needs_training_on"]    = train_df["needs_training_on"].fillna("")

    # Simple rollup
    gap_counts = (train_df["specialized_training"].str.lower()
                  .isin(["no","partial"])).groupby(train_df["role"]).sum().rename("gaps")
    by_role = pd.DataFrame({"gaps": gap_counts}).fillna(0).astype(int).sort_values("gaps", ascending=False)
    st.dataframe(by_role, use_container_width=True)

    # Isolates
    iso = [n for n in G.nodes() if G.in_degree(n)==0 and G.out_degree(n)==0]
    if iso:
        iso_df = st.session_state["nodes"][st.session_state["nodes"]["id"].isin(iso)][["id","label","role"]]
        st.warning("Isolated nodes (not connected):")
        st.dataframe(iso_df, use_container_width=True)
    else:
        iso_df = pd.DataFrame(columns=["id","label","role"])
        st.success("No isolated nodes.")

    use_sart_mustlinks = st.checkbox("Use SART must-link set", value=True)
    
    # Must-link gaps
    gaps_df = must_link_gaps(st.session_state["nodes"], st.session_state["edges"]) if use_sart_mustlinks else pd.DataFrame()
    if len(gaps_df):
        st.error("Must-link gaps (either role missing or no path):")
        st.dataframe(gaps_df, use_container_width=True)
        st.caption("Consider adding warm-handoffs or MOUs to cover these pathways.")
    else:
        st.success("All must-link pathways present.")

st.markdown("### Readiness Meter")
colL, colR = st.columns([2,1])
with colL:
    st.caption("Foundational docs & structures")
    c1 = st.checkbox("Written charter/mission", value=False, key="rm_charter")
    c2 = st.checkbox("Core protocols documented", value=False, key="rm_protocols")
    c3 = st.checkbox("MOUs in place (key partners)", value=False, key="rm_mous")
    c4 = st.checkbox("Case-review cadence set", value=False, key="rm_case_review")
    c5 = st.checkbox("All core roles present (Advocate, SANE, LE, Prosecutor, Lab, ED)", value=False, key="rm_roles")

    st.caption("Victim-centered timing (typical, not max)")
    t_adv = st.number_input("Hours to advocate contact", min_value=0.0, value=2.0, step=0.5, key="rm_t_adv")
    t_exam= st.number_input("Hours to SANE exam",       min_value=0.0, value=24.0, step=1.0, key="rm_t_exam")
    t_lab = st.number_input("Hours to lab submission",  min_value=0.0, value=48.0, step=1.0, key="rm_t_lab")

checks  = {
    "has_charter": st.session_state["rm_charter"],
    "has_protocols": st.session_state["rm_protocols"],
    "has_mous": st.session_state["rm_mous"],
    "case_review_cadence": st.session_state["rm_case_review"],
    "all_core_roles_present": st.session_state["rm_roles"],
}
metrics = {"hours_to_advocate": t_adv, "hours_to_exam": t_exam, "hours_to_lab": t_lab}
status, detail = readiness_score(checks, metrics)

with colR:
    badge = {"green":"🟢","yellow":"🟡","red":"🔴"}[status]
    st.markdown(f"#### Status: {badge}  \nScore: **{detail['score']}** / 10")
    st.caption(f"Docs {detail['doc_ok']}/4 • SLAs {detail['sla_hits']}/3 • Core roles {'✅' if detail['part_ok'] else '❌'}")

    # (Optional) stash for Export Kit if you already built it
    st.session_state["readiness_status"] = status
    st.session_state["readiness_score"]  = detail["score"]

    # --- Export Kit controls ---
    st.markdown("### Export Kit")
    decisions = st.text_area("Facilitator notes (decisions / next steps)", height=120, placeholder="E.g., Finalize ED→Advocacy on-call MOU; Reserve 3 DV-priority housing slots; Add firearm surrender workflow…")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Build Export Kit (XLSX)"):
            xlsx = export_kit_xlsx(
                st.session_state["nodes"], st.session_state["edges"],
                cent, iso, gaps_df, MUST_LINKS,
                decisions_text=decisions,
                readiness_status=st.session_state.get("readiness_status"),
                readiness_score=st.session_state.get("readiness_score"),
                readiness_missing={
                    "Written charter/mission": st.session_state.get("rm_charter", False),
                    "Core protocols documented": st.session_state.get("rm_protocols", False),
                    "MOUs in place (key partners)": st.session_state.get("rm_mous", False),
                    "Case-review cadence set": st.session_state.get("rm_case_review", False),
                    "All core roles present": st.session_state.get("rm_roles", False),
                    # SLAs (you can adjust thresholds in readiness_score)
                    "Hours to advocate ≤ 2": (st.session_state.get("rm_t_adv", 999) <= 2),
                    "Hours to SANE exam ≤ 24": (st.session_state.get("rm_t_exam", 999) <= 24),
                    "Hours to lab submission ≤ 48": (st.session_state.get("rm_t_lab", 999) <= 48),
                }
            )
            st.download_button(
                "Download Export Kit (.xlsx)",
                data=xlsx,
                file_name="OVW_PA18_Export_Kit.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    with col2:
        html = export_kit_html(
            st.session_state["nodes"], st.session_state["edges"],
            cent, iso, gaps_df, MUST_LINKS,
            brand_png_path=str(LOGO_MARK) if 'LOGO_MARK' in globals() else None,
            decisions_text=decisions,
            readiness_status=st.session_state.get("readiness_status"),
            readiness_score=st.session_state.get("readiness_score"),
            readiness_missing={
                "Written charter/mission": st.session_state.get("rm_charter", False),
                "Core protocols documented": st.session_state.get("rm_protocols", False),
                "MOUs in place (key partners)": st.session_state.get("rm_mous", False),
                "Case-review cadence set": st.session_state.get("rm_case_review", False),
                "All core roles present": st.session_state.get("rm_roles", False),
                "Hours to advocate ≤ 2": (st.session_state.get("rm_t_adv", 999) <= 2),
                "Hours to SANE exam ≤ 24": (st.session_state.get("rm_t_exam", 999) <= 24),
                "Hours to lab submission ≤ 48": (st.session_state.get("rm_t_lab", 999) <= 48),
            }
        )
        st.download_button(
            "Download One-Pager (HTML)",
            data=html,
            file_name="OVW_PA18_Export_Kit.html",
            mime="text/html"
        )
        st.caption("Tip: open the HTML and print to PDF.")

with tab_template:
    st.markdown("### Role & Intercept Reference")
    st.write("**Roles**:", ", ".join(DV_ROLES))
    st.write("**Intercepts**:", ", ".join(DV_INTERCEPTS))
    st.caption("Tip: keep node labels human-friendly (what people call them locally). Use role to classify.")

