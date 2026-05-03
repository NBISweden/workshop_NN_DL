import nbformat
import sys
import re

MARKER_HIDDEN = "# source_hidden"
MARKER_SOLUTION = "# solution_hidden"
SEPARATORS    = {"<!-- -->", "<!-- cell -->", ""}   # cells to remove

def process_notebook(nb):
    new_cells = []
    for cell in nb.cells:
        source = "".join([x.strip() for x in cell.source])
        if (cell.cell_type == "raw") and (cell.source in SEPARATORS):
            continue
        if cell.source.startswith(MARKER_HIDDEN):
            cell.metadata["jupyter"] = {"source_hidden": True}
        if cell.source.startswith(MARKER_SOLUTION):
            cell.metadata["jupyter"] = {"source_hidden": True}
        new_cells.append(cell)
    nb.cells = new_cells
    return nb

with open(sys.argv[1]) as f:
    nb = nbformat.read(f, as_version=4)

nb = process_notebook(nb)

with open(sys.argv[1], "w") as f:
    nbformat.write(nb, f)
