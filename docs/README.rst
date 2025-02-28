usv-playpen v0.8.0
==================

GUI to facilitate conducting experiments with multi-probe e-phys (Neuropixels), multi-channel audio (Avisoft) and multi-camera video (Loopbio) acquisition. Developed for behavioral recording purposes at the `Princeton Neuroscience Institute <https://pni.princeton.edu/>`_ 2021-25 (Falkner/Murthy labs). Due to proprietary software design and limitations, recordings can only be performed on OS Windows. The data processing, analysis and visualization branches of the GUI are platform-independent.

.. image:: https://img.shields.io/badge/Python-3.10-blue
   :target: https://img.shields.io/badge/Python-3.10-blue

.. image:: https://zenodo.org/badge/566588932.svg
   :target: https://zenodo.org/badge/latestdoi/566588932

.. image:: https://img.shields.io/github/repo-size/bartulem/usv-playpen
   :target: https://github.com/bartulem/usv-playpen/

.. image:: https://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active

.. image:: https://img.shields.io/github/issues/bartulem/usv-playpen
   :target: https://github.com/bartulem/usv-playpen

.. image:: https://img.shields.io/github/license/bartulem/usv-playpen
   :target: https://github.com/bartulem/usv-playpen/blob/main/LICENSE

.. image:: https://img.shields.io/github/stars/bartulem/usv-playpen?style=social
   :target: https://github.com/bartulem/usv-playpen/

.. image:: https://img.shields.io/github/forks/bartulem/usv-playpen?style=social
   :target: https://github.com/bartulem/usv-playpen/

.. image:: https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square
   :target: https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square

Prerequisites
-------------

* `Helvetica <https://freefontsfamily.net/helvetica-font-family/>`_ (download and install)
* `Anaconda <https://www.anaconda.com/download>`_ (and add it to top of PATH)
* `git <https://git-scm.com/download/>`_ (and add it to PATH)
* `ffmpeg <https://ffmpeg.org/download.html>`_ (and add it to PATH)
* `sox <https://sourceforge.net/projects/sox/>`_ (and add it to PATH)
* `sleap <https://sleap.ai/>`_
* `das <https://janclemenslab.org/das/>`_
* `CoolTerm <https://coolterm.en.lo4d.com/windows>`_

Installation and updating
-------------------------

Set up a new conda environment with Python 3.10 and give it any name, *e.g.*, usv.

.. code-block:: bash

   conda create --name usv python=3.10 -c conda-forge -y

Activate the virtual environment with:

.. code-block:: bash

   conda activate usv

Install GUI with command below. Also, rerun the same command to check for and install updates.

.. code-block:: bash

   pip install git+https://github.com/bartulem/usv-playpen#egg=usv-playpen --use-pep517

Add the python-motifapi package to your virtual environment:

.. code-block:: bash

   pip install git+https://github.com/loopbio/python-motifapi.git#egg=motifapi --use-pep517

Test installation
-----------------

At the end of each command for testing, make sure you add a valid e-mail address a test e-mail can be sent to.

.. code-block:: bash

   conda activate usv
   python -m usv_playpen._tests.test_recording username@domain.com
   python -m usv_playpen._tests.test_processing username@domain.com

Usage
-----

Load the environment with the appropriate name, *e.g.*, usv., and run the GUI:

.. code-block:: bash

   conda activate usv && usv-playpen

Developed in PyCharm Pro 2024.3, and tested on macOS Sequoia 15.1 / Pop!_OS 22.04 / Windows 11.