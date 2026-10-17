import nbformat
from nbclient import NotebookClient
nb = nbformat.read("autoregressive_textgen.ipynb", as_version=4)
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, "autoregressive_textgen.ipynb")
