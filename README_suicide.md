# Systems Mapping (Suicide Prevention)

This repository contains two Streamlit apps:
- `app_suicide.py` — **current** suicide-prevention systems mapping
- `app_domestic_violence.py` — archived DV/SA version (formerly `app.py`)

## What’s new
- Suicide-prevention taxonomy, vignettes, and starter nodes
- Updated defaults and categories in `app_streamlit.py`
- Cleaned terminology throughout (prevention, crisis, postvention, lethal means safety)

## Quick start

```bash
# install environment if needed
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate
pip install -r requirements.txt

# run the suicide-prevention app
streamlit run app_suicide.py
```

## Data templates
- `data/templates/taxonomy_template.json`
- `data/templates/vignettes_template.json`
- `data/templates/nodes_template.csv`

These ship with suicide-prevention defaults; you can replace them with local data to fit your community context.

## Purpose
The Systems Mapping platform helps communities understand, visualize, and strengthen suicide-prevention systems across prevention, intervention, treatment, and postvention supports. It can be adapted for use with other behavioral health or crisis system planning contexts.
