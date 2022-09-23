Build zeugmas from the magpie corpus
====================================

How to see this:

(do this in a virtualenv)

```console
pip install -U -r requirements.txt
jupyter nbextension install jupytext --py --sys-prefix
jupyter nbextension enable jupytext --py --sys-prefix
jupyter notebook magpie.md
```

Then run all cells.
