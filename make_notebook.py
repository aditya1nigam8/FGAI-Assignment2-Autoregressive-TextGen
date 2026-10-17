import re, nbformat as nbf
src = open("autoregressive_textgen.py", encoding="utf-8").read()
cells = []
for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
    if chunk.startswith(" [markdown]"):
        body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in chunk.split("\n")[1:]).strip()
        cells.append(nbf.v4.new_markdown_cell(body))
    else:
        cells.append(nbf.v4.new_code_cell(chunk.strip("\n")))
nb = nbf.v4.new_notebook(cells=cells)
nbf.write(nb, "autoregressive_textgen.ipynb")
