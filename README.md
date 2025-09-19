[![Python version](https://img.shields.io/badge/Python-3.10-blue)](https://img.shields.io/badge/Python-3.10-blue)
[![DOI](https://zenodo.org/badge/566588932.svg)](https://zenodo.org/badge/latestdoi/566588932)
[![repo size](https://img.shields.io/github/repo-size/bartulem/usv-playpen)](https://github.com/bartulem/usv-playpen/)
[![Documentation Status](https://readthedocs.org/projects//usv-playpen/badge/?version=latest)](https://usv-playpen.readthedocs.io/en/latest/?badge=latest)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![shields.io-issues](https://img.shields.io/github/issues/bartulem/usv-playpen)
[![Release](https://img.shields.io/github/v/release/bartulem/usv-playpen)](https://img.shields.io/github/v/release/bartulem/usv-playpen)
[![MIT Licence](https://img.shields.io/github/license/bartulem/usv-playpen)](https://github.com/bartulem/usv-playpen/blob/main/LICENSE)
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)
[![GitHub stars](https://img.shields.io/github/stars/bartulem/usv-playpen?style=social)](https://github.com/bartulem/usv-playpen/)
[![GitHub forks](https://img.shields.io/github/forks/bartulem/usv-playpen?style=social)](https://github.com/bartulem/usv-playpen/)

# usv-playpen

![](https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/src/usv_playpen/img/usv_playpen_gui.png)

GUI/CLI to facilitate conducting experiments with multi-probe e-phys
(Neuropixels), multichannel audio (Avisoft) and multi-camera video (Loopbio)
acquisition. Developed for behavioral recording purposes at the
[Princeton Neuroscience Institute](https://pni.princeton.edu/) 2021-26
([Falkner](https://www.falknerlab.com/)/[Murthy](https://murthylab.princeton.edu/)
labs). Due to proprietary software design and limitations, recordings can only
be performed on OS Windows. The data processing, analysis and visualization
branches of the GUI are platform-independent.

## Prerequisites

- [CoolTerm](https://coolterm.en.lo4d.com/windows) (necessary only on the audio
  recording PC)
- [git](https://git-scm.com/download/)  (if on Windows, add PATH to USER
  VARIABLES)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (if on Windows, add PATH to USER
  VARIABLES)
- [sleap](https://sleap.ai/) (install in standalone conda environment)
- [das](https://janclemenslab.org/das/) (install in standalone conda
  environment)
- [vocalocator-ssl](https://github.com/Aramist/vocalocator-ssl) (install in
  standalone conda environment)

## Installation and updating

Clone the repository and set up virtual environment with *uv*:

```bash
git clone https://github.com/bartulem/usv-playpen.git
cd usv-playpen
uv venv --python=3.10
```

### Linux (terminal) instructions

```bash
echo 'alias activate-pni="source /path/.../usv-playpen/.venv/bin/activate"' >> ~/.bashrc
source ~/.bashrc
activate-pni
uv sync
```

### macOS (terminal) instructions

```bash
echo 'alias activate-pni="source /path/.../usv-playpen/.venv/bin/activate"' >> ~/.zshrc
source ~/.zshrc
activate-pni
uv sync
```

### Windows (powershell) instructions

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
Add-Content -Path $PROFILE -Value "`nfunction activate-pni {`n    . 'C:\path\...\usv-playpen\.venv\Scripts\Activate.ps1'`n}"
. $PROFILE
activate-pni
uv sync
```

Navigate to the cloned repository and use the following command to check for and install updates:

```bash
git pull && uv sync
```

## Usage

Run the GUI with:

```bash
activate-pni && usv-playpen
```

User guide with detailed instructions is available
[here](https://usv-playpen.readthedocs.io/en/latest/).
