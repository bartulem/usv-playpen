"""
@author: bartulem
Installed-metadata guard for the package version.

Coverage: the version recorded in the installed distribution metadata equals
``usv_playpen.__version__``, so a stale editable install or a mis-generated
``_version.py`` cannot ship a build that reports the wrong release.
"""

from __future__ import annotations

import importlib.metadata

import usv_playpen as m


def test_version():
    """
    Description
    -----------
    Asserts that the version in the installed distribution metadata
    (``importlib.metadata.version``) equals ``usv_playpen.__version__``. The two
    diverge when an editable install is stale or ``_version.py`` was generated
    from the wrong tag, and then a release, a bug report and ``pip show``
    disagree about which code is running.

    Parameters
    ----------

    Returns
    -------
    None
    """

    assert importlib.metadata.version("usv_playpen") == m.__version__
