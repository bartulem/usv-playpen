# Download README from main branch when building documentation, following:
# https://stackoverflow.com/questions/66495200/is-it-possible-to-include-external-rst-files-in-my-documentation
from urllib.request import urlretrieve
urlretrieve (
    url = "https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/README.md",
    filename = "README.md"
)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'usv-playpen'
copyright = '2025, github/bartulem'
author = 'Bartul Mimica (documentation)'
release = '0.8.6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'myst_parser',
    'nbsphinx',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_dark_mode'
    ]

# Notebooks will be displayed even if they include errors
nbsphinx_allow_errors = True
# Don't auto-execute notebooks.
nbsphinx_execute = 'never'
# Set to True if you want dark mode to be the default for first-time visitors.
default_dark_mode = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_logo = 'media/gui_icon.png'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'canonical_url': '',
    'logo_only': False,
    'prev_next_buttons_location': 'top',
    'style_external_links': False,
    'style_nav_header_background': 'Gray',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
