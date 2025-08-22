# usv-playpen v0.8.10

![](https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/src/img/usv_playpen_gui.png)

GUI/CLI to facilitate conducting experiments with multi-probe e-phys (Neuropixels), multichannel audio (Avisoft) and multi-camera video (Loopbio) acquisition. Developed for behavioral recording purposes at the [Princeton Neuroscience Institute](https://pni.princeton.edu/) 2021-26 ([Falkner](https://www.falknerlab.com/)/[Murthy](https://murthylab.princeton.edu/) labs). Due to proprietary software design and limitations, recordings can only be performed on OS Windows. The data processing, analysis and visualization branches of the GUI are platform-independent.

[![Python version](https://img.shields.io/badge/Python-3.10-blue)](https://img.shields.io/badge/Python-3.10-blue)
[![DOI](https://zenodo.org/badge/566588932.svg)](https://zenodo.org/badge/latestdoi/566588932)
[![repo size](https://img.shields.io/github/repo-size/bartulem/usv-playpen)](https://github.com/bartulem/usv-playpen/)
[![Documentation Status](https://readthedocs.org/projects//usv-playpen/badge/?version=latest)](https://usv-playpen.readthedocs.io/en/latest/?badge=latest)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![shields.io-issues](https://img.shields.io/github/issues/bartulem/usv-playpen)
[![Release](https://img.shields.io/github/v/release/bartulem/usv-playpen)](https://img.shields.io/github/v/release/bartulem/usv-playpen)
[![MIT Licence](https://img.shields.io/github/license/bartulem/usv-playpen)](https://github.com/bartulem/usv-playpen/blob/main/LICENSE)
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)
[![GitHub stars](https://img.shields.io/github/stars/bartulem/usv-playpen?style=social)](https://github.com/bartulem/usv-playpen/)
[![GitHub forks](https://img.shields.io/github/forks/bartulem/usv-playpen?style=social)](https://github.com/bartulem/usv-playpen/)


## Prerequisites

* [CoolTerm](https://coolterm.en.lo4d.com/windows) (necessary only on the audio recording PC)
* [Anaconda](https://www.anaconda.com/download) (if on Windows, add PATH to USER VARIABLES)
* [sleap](https://sleap.ai/) (install in standalone conda environment)
* [das](https://janclemenslab.org/das/) (install in standalone conda environment)
* [vocalocator-ssl](https://github.com/Aramist/vocalocator-ssl) (install in standalone conda environment)

## Installation and updating

Set up a new conda environment with Python 3.10 and give it any name, _e.g._, pni.
```bash
conda create -n pni python=3.10 ffmpeg git sox -c conda-forge -y
```
Activate the virtual environment with:
```bash
conda activate pni
```
Install GUI with command below. Also, rerun the same command to check for and install updates.
```bash
pip install git+https://github.com/bartulem/usv-playpen --use-pep517
```
Add the python-motifapi package to your virtual environment:
```bash
pip install git+https://github.com/loopbio/python-motifapi --use-pep517
```

## Usage

Load the environment with the appropriate name, _e.g._, pni, and run the GUI:
```bash
conda activate pni && usv-playpen
```

User guide with detailed instructions is available [here](https://usv-playpen.readthedocs.io/en/latest/).