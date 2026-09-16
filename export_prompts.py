"""Mechanically export the two prompts and hashes from the included source."""
import hashlib
import json
from pathlib import Path

from trata_slime_muse import turn_credit_synthesis_spans as source

ROOT = Path(__file__).resolve().parent


def export():
    for name, prompt in [("objective", source.OBJECTIVE_JUDGE_SYSTEM_PROMPT),
                         ("attribution", source.ATTRIBUTION_JUDGE_SYSTEM_PROMPT)]:
        (ROOT / "prompts" / f"{name}.md").write_text(prompt + "\n")
    paths = ["trata_slime_muse/turn_credit_synthesis_spans.py",
             "trata_slime_muse/turn_credit_native.py"]
    manifest = {
        "description": "Unmodified source-module snapshot; no training artifacts included.",
        "prompt_version": source.PROMPT_VERSION,
        "schema_version": source.SCHEMA_VERSION,
        "files": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}
                  for p in paths],
    }
    (ROOT / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    export()
