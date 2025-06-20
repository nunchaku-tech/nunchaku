# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path

project = "Nunchaku"
copyright = "2025, Nunchaku Team"
author = "Nunchaku Team"

version_path = Path(__file__).parent.parent.parent / "nunchaku" / "__version__.py"
version_ns = {}
exec(version_path.read_text(), {}, version_ns)
version = release = version_ns["__version__"]
# release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "breathe",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Include global link definitions -----------------------------------------
rst_prolog = """
.. include:: links.rst
"""


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_permalinks_icon = "<span>#</span>"
# html_theme = "sphinxawesome_theme"
html_theme = "furo"
html_static_path = ["_static"]
