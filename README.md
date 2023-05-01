# usv-playpen v0.1.4

GUI to facilitate conducting experiments with multichannel audio (Avisoft) and multi-camera video (Loopbio) acquisition. Developed for behavioral recording purposes at the [Princeton Neuroscience Institute](https://pni.princeton.edu/) 2021-23 (Falkner/Murthy labs). Due to proprietary software design and limitations, recordings can only be performed on OS Windows. The data processing branch of the GUI is platform-independent.

<p align="center">
  <img src="src/img/usv_playpen_gui.png">
</p>

## Prerequisites

* [Python 3.10](https://www.python.org/downloads/) (and add it to top of PATH)
* [pip](https://pip.pypa.io/en/stable/) (and add it to PATH)
* [git](https://git-scm.com/download/)  (and add it to PATH)

## Installation and updating

Set up a new virtual environment with Python 3.10 and give it any name, _e.g._, venv_name.
```bash
python -m venv venv_name
```
Activate the virtual environment with:
```bash
source ./venv_name/bin/activate
```
or, on OS Windows:
```bash
.\venv_name\Scripts\activate
```
Install GUI with command below. Also, rerun the same command to check for and install updates.
```bash
pip install git+https://github.com/bartulem/usv-playpen --use-pep517
```

Add the python-motifapi package to your virtual environment:
```bash
pip install git+https://github.com/loopbio/python-motifapi.git#egg=motifapi --use-pep517
```

## Features

* behavioral_experiments.ExperimentController --> run behavioral experiments with Loopbio/Avisoft software
* extract_phidget_data.Gatherer --> extract data measured by illumination and temperature/humidity phidgets
* file_manipulation.Operator --> (1) break from multi to single channel, band-pass filter and temporally concatenate Avisoft-generated audio (_e.g._, WAV) files,
                                 (2) concatenate Motif-generated video (_e.g._, mp4) files and rectify their frame rates (fps)
* synchronize_files.Synchronizer --> cut WAV file to video file (_e.g._, mp4) length and perform a/v synchronization check
* preprocessing_plots.SummaryPlotter --> generate summary figure for data preprocessing (_i.e._, metadata and sync quality)

## Usage

Locate the pip installed package:
```bash
pip show usv-playpen
```
Navigate to the directory w/ the "usv_playpen_gui.py" file (example path listed below).
```bash
cd /.../venv_name/lib/site-packages/usv-playpen
```

Run the GUI.
```bash
python usv_playpen_gui.py
```

Developed and tested in PyCharm Pro 2023.1.1, on Windows 10/Ubuntu 22.04 LTS.
