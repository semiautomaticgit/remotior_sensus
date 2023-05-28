# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))

# -- Project information -----------------------------------------------------

project = 'Remotior Sensus'
copyright = '2022-2023, Luca Congedo'
author = 'Luca Congedo'
release = '0.0.35'
version = '0.0.35.1'


# -- General configuration ---------------------------------------------------

master_doc = 'index'
locale_dirs = ['locale/']
gettext_compact = False
extensions = ['sphinx.ext.napoleon', 'sphinx.ext.doctest']
templates_path = ['_templates']
exclude_patterns = ['modules.rst', 'remotior_sensus.util.*']
autodoc_default_options = {
    'special-members': '__init__',
}
pygments_style = 'github-dark'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_show_sourcelink = False
html_static_path = ['_static']
html_favicon = "_static/favicon.ico"
html_show_sphinx = False

# -- Options for LaTeX output ---------------------------------------------

latex_documents = [
  ('index', 'Remotior_Sensus.tex', 'Remotior Sensus Documentation',
   'Luca Congedo', 'manual'),
]
latex_logo = '_static/logo.png'
latex_use_parts = True
latex_show_pagerefs = True
latex_elements = {
    'papersize': 'a4paper',
    'sphinxsetup': """
        pre_box-shadow=2pt 2pt,
    """
}
# -- Options for manual page output ---------------------------------------

man_pages = [
    ('index', 'remotior_sensus', 'Remotior Sensus Documentation',
     ['Luca Congedo'], 1)
]

# -- Options for Texinfo output -------------------------------------------

texinfo_documents = [
  ('index', 'Remotior_Sensus', 'Remotior Sensus Documentation',
   'Luca Congedo', 'Remotior_Sensus', 
   'Software to process remote sensing and GIS data.',
   'GIS and Remote Sensing'),
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
