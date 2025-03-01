# usv-playpen v0.8.0

![](https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/src/img/usv_playpen_gui.png)

GUI to facilitate conducting experiments with multi-probe e-phys (Neuropixels), multi-channel audio (Avisoft) and multi-camera video (Loopbio) acquisition. Developed for behavioral recording purposes at the [Princeton Neuroscience Institute](https://pni.princeton.edu/) 2021-25 (Falkner/Murthy labs). Due to proprietary software design and limitations, recordings can only be performed on OS Windows. The data processing, analysis and visualization branches of the GUI are platform-independent.

[![Python version](https://img.shields.io/badge/Python-3.10-blue)](https://img.shields.io/badge/Python-3.10-blue)
[![DOI](https://zenodo.org/badge/566588932.svg)](https://zenodo.org/badge/latestdoi/566588932)
[![repo size](https://img.shields.io/github/repo-size/bartulem/usv-playpen)](https://github.com/bartulem/usv-playpen/)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![shields.io-issues](https://img.shields.io/github/issues/bartulem/usv-playpen)
[![MIT Licence](https://img.shields.io/github/license/bartulem/usv-playpen)](https://github.com/bartulem/usv-playpen/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/bartulem/usv-playpen?style=social)](https://github.com/bartulem/usv-playpen/)
[![GitHub forks](https://img.shields.io/github/forks/bartulem/usv-playpen?style=social)](https://github.com/bartulem/usv-playpen/)
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)


## Prerequisites

* [Helvetica](https://freefontsfamily.net/helvetica-font-family/) (download and install)
* [Anaconda](https://www.anaconda.com/download) (and add it to top of PATH)
* [git](https://git-scm.com/download/)  (and add it to PATH)
* [ffmpeg](https://ffmpeg.org/download.html) (and add it to PATH)
* [sox](https://sourceforge.net/projects/sox/)  (and add it to PATH)
* [sleap](https://sleap.ai/) 
* [das](https://janclemenslab.org/das/)
* [CoolTerm](https://coolterm.en.lo4d.com/windows)

## Installation and updating

Set up a new conda environment with Python 3.10 and give it any name, _e.g._, usv.
```bash
conda create --name usv python=3.10 -c conda-forge -y
```
Activate the virtual environment with:
```bash
conda activate usv
```
Install GUI with command below. Also, rerun the same command to check for and install updates.
```bash
pip install git+https://github.com/bartulem/usv-playpen#egg=usv-playpen --use-pep517
```

Add the python-motifapi package to your virtual environment:
```bash
pip install git+https://github.com/loopbio/python-motifapi.git#egg=motifapi --use-pep517
```

## Test installation

At the end of each command for testing, make sure you add a valid e-mail address a test e-mail can be sent to.
```bash
conda activate usv
python -m usv_playpen._tests.test_recording username@domain.com
python -m usv_playpen._tests.test_processing username@domain.com
```

## Usage

Load the environment with the appropriate name, _e.g._, usv., and run the GUI:
```bash
conda activate usv && usv-playpen
```

User guide with detaile instructions is available [here](https://usv-playpen.readthedocs.io/en/latest/).