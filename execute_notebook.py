"""Execute Project 1 and save all verified outputs into the notebook."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


project_dir = Path(__file__).resolve().parent
notebook_path = project_dir / "notebooks" / "Project_1_Edmunds_Car_Recommender_Tarun.ipynb"

with notebook_path.open("r", encoding="utf-8") as handle:
    notebook = nbformat.read(handle, as_version=4)

client = NotebookClient(
    notebook,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": str(notebook_path.parent)}},
    allow_errors=False,
)
client.execute()

with notebook_path.open("w", encoding="utf-8") as handle:
    nbformat.write(notebook, handle)

print(f"Executed and saved {notebook_path}")
