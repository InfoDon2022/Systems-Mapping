# validate_vignettes.py
# Usage:
#   python validate_vignettes.py path/to/vignettes.json schemas/vignette.schema.json
# Requires: jsonschema  (pip install jsonschema)

import json
import sys
from jsonschema import validate, Draft7Validator
from jsonschema.exceptions import ValidationError

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main(vignettes_path: str, schema_path: str):
    data = load_json(vignettes_path)
    schema = load_json(schema_path)

    # Pre-flight: show all errors, not just the first
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)

    if not errors:
        print("✅ Vignette file is valid.")
        # Optional: basic summary
        vignettes = data.get("vignettes", [])
        print(f"Found {len(vignettes)} vignette(s). IDs:", [v.get("id") for v in vignettes])
        sys.exit(0)
    else:
        print("❌ Vignette file is INVALID. Details:")
        for err in errors:
            # Build a readable path like vignettes[0].title
            path = "root"
            for p in err.path:
                if isinstance(p, int):
                    path += f"[{p}]"
                else:
                    path += f".{p}"
            print(f"- at {path}: {err.message}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_vignettes.py <vignettes.json> <schema.json>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
