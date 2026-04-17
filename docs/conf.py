import os
import sys

# Make the package importable when building on Read the Docs
sys.path.insert(0, os.path.abspath(".."))

# --- Project information ---

project = "simGL"
author = "Moisès Coll Macià, Graham Gower"
copyright = "2024, Moisès Coll Macià, Graham Gower"
release = "0.2.0"

# --- General configuration ---

extensions = [
    "sphinx.ext.napoleon",     # NumPy / Google docstring support
    "autoapi.extension",       # Auto-generate API docs from source
    "sphinx.ext.viewcode",     # Add [source] links to API pages
    "sphinx.ext.intersphinx",  # Cross-reference numpy / python docs
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# --- AutoAPI ---

autoapi_dirs = ["../simGL"]
autoapi_ignore = ["*/.ipynb_checkpoints/*"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
suppress_warnings = ["autoapi.python_import_resolution"]

# --- Napoleon (docstring style) ---

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_returns = True

# --- Intersphinx ---

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# --- HTML output ---

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
}
