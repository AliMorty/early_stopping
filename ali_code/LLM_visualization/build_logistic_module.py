from pathlib import Path
import nbformat
from nbconvert import PythonExporter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # ali_code/ (Ali's code); this script lives in ali_code/LLM_visualization/
NB = ROOT / "logistic_regression_hold_out.ipynb"
OUT = ROOT / "logistic_regression.py"

nb = nbformat.read(str(NB), as_version=4)
nb.cells = [c for c in nb.cells
            if c.cell_type == "code" and "module" in c.get("metadata", {}).get("tags", [])]
code, _ = PythonExporter().from_notebook_node(nb)
with open(OUT, "w") as f:
    f.write("# AUTO-GENERATED from logistic_regression_hold_out.ipynb — do not edit.\n")
    f.write("# Regenerate: python ali_code/LLM_visualization/build_logistic_module.py\n")
    f.write(code)
print(f"wrote {OUT}")
