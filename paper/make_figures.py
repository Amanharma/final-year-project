"""Copy and prepare figures for paper/main.tex.

Copies result figures from results/ into paper/figures/ and generates
any derived publication-ready figures. Run from the project root or
from inside the paper/ directory.
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "results")
DST  = os.path.join(ROOT, "paper", "figures")

os.makedirs(DST, exist_ok=True)

COPIES = {
    "physics_twin.png":        "physics_twin.png",
    "qml_fair.png":            "qml_fair.png",
    "qaoa_scaling.png":        "qaoa_scaling.png",
    "fault_sweep.png":         "fault_sweep.png",
    "qml_comparison.png":      "qml_comparison.png",
}

missing = []
for src_name, dst_name in COPIES.items():
    src_path = os.path.join(SRC, src_name)
    dst_path = os.path.join(DST, dst_name)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"  copied {src_name} -> figures/{dst_name}")
    else:
        print(f"  WARNING: {src_path} not found, skipping", file=sys.stderr)
        missing.append(src_name)

if missing:
    print(f"\nMissing {len(missing)} figure(s): {missing}", file=sys.stderr)
    print("Run the relevant experiment scripts first.", file=sys.stderr)
else:
    print("\nAll figures ready in paper/figures/")
