# Download README from main branch when building documentation, following:
# https://stackoverflow.com/questions/66495200/is-it-possible-to-include-external-rst-files-in-my-documentation
from urllib.request import urlretrieve
urlretrieve (
    "https://github.com/bartulem/usv-playpen/blob/main/README.md",
    "README.md"
)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'usv-playpen'
copyright = '2025, Bartul Mimica'
author = 'Bartul Mimica (documentation)'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'myst_parser',            # Use markdown files in addition to .rst
    'nbsphinx',               # Render notebooks
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
    ]

autoapi_dirs = ['.../usv-playpen']
master_doc = 'index'

# Notebooks will be displayed even if they include errors
nbsphinx_allow_errors = True
# Don't auto-execute notebooks.
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_logo = 'https://github.com/bartulem/usv-playpen/blob/main/src/img/gui_icon.png'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'canonical_url': '',
    'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'top',
    'style_external_links': False,
    'style_nav_header_background': 'Gray',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
html_static_path = ['_static']