# Creating a combined .ipynb that merges the existing notebook and embeds the final PDF report.
# This code will:
# 1. Read the existing notebook at /mnt/data/answer 5 eda1.ipynb
# 2. Create a new notebook that contains:
#    - an introductory Markdown cell
#    - all cells from the original notebook (preserving order)
#    - a Markdown cell noting the embedded PDF (final report)
#    - a Code cell that displays the PDF inline using an IFrame
# 3. Save the combined notebook to /mnt/data/answer_5_combined.ipynb
# After execution, this cell will print the path to the combined notebook.
# Note: The embedded PDF file is /mnt/data/answers final EDA1.pdf (sourced in this notebook).

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

orig_nb_path = '/mnt/data/answer 5 eda1.ipynb'
pdf_filename = '/mnt/data/answers final EDA1.pdf'
combined_nb_path = '/mnt/data/answer_5_combined.ipynb'

# Read original notebook
if not os.path.exists(orig_nb_path):
    raise FileNotFoundError(f"Original notebook not found at: {orig_nb_path}")

orig_nb = nbformat.read(orig_nb_path, as_version=4)

# Create new notebook and populate
nb = new_notebook()
intro_md = (
    "# Combined EDA Notebook\n\n"
    "This notebook combines your original analysis notebook (`answer 5 eda1.ipynb`) with the final\n"
    "written report embedded below as a PDF. The PDF was provided as `answers final EDA1.pdf`. \n\n"
    "**Source PDF (final report)**: referenced and embedded for convenient review. fileciteturn7file4\n\n"
    "---\n\n"
    "### Contents\n"
    "1. Original notebook cells (code, outputs, and markdown) — preserved in order.\n"
    "2. Embedded final report (PDF) displayed below.\n\n"
    "_If you want the PDF text converted into Markdown cells instead of an embedded PDF, say the word and I'll do that next._\n"
)
nb.cells.append(new_markdown_cell(intro_md))

# Append all cells from original notebook
for cell in orig_nb.cells:
    nb.cells.append(cell)

# Add a separator markdown and a code cell to display the PDF
report_md = "## Final report (embedded PDF)\nThe PDF is embedded below for quick inspection. It is the same file used as the 'answers final EDA1.pdf' report. fileciteturn7file4"
nb.cells.append(new_markdown_cell(report_md))

display_code = (
    "from IPython.display import IFrame, display\n"
    "import os\n\n"
    "pdf_path = 'answers final EDA1.pdf'\n"
    "if os.path.exists(pdf_path):\n"
    "    display(IFrame(pdf_path, width=900, height=700))\n"
    "else:\n"
    "    print('PDF not found at', pdf_path)\n"
)
nb.cells.append(new_code_cell(display_code))

# Save combined notebook
nbformat.write(nb, combined_nb_path)
combined_nb_path

