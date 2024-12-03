"""
@author: bartulem
Configure path to the OS in use.
"""

import platform


def configure_path(pa):
    """
    Description
    ----------
    This function converts path names between OSs.
    ----------

    Parameter
    ---------
    pa : str
        Original path.

    Returns
    -------
     pa : str
        OS-converted path.
    """

    if pa.startswith('F:\\'):
        if platform.system() == 'Darwin':
            pa = pa.replace('\\', '/').replace('F:', '/Volumes/falkner')
        elif platform.system() == 'Linux':
            pa = pa.replace('\\', '/').replace('F:', '/mnt/falkner')
    else:
        if pa.startswith('/mnt'):
            if platform.system() == 'Windows':
                pa = pa.replace('/mnt/falkner', 'F:').replace('/', '\\')
            elif platform.system() == 'Darwin':
                pa = pa.replace('mnt', 'Volumes')
        elif pa.startswith('/Volumes'):
            if platform.system() == 'Windows':
                pa = pa.replace('/Volumes/falkner', 'F:').replace('/', '\\')
            elif platform.system() == 'Linux':
                pa = pa.replace('Volumes', 'mnt')

    return pa
