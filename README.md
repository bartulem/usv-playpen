[![Python version](https://img.shields.io/badge/Python-3.13%20|%203.14-blue)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.7926971-blue)](https://doi.org/10.5281/zenodo.7926971)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue)](https://github.com/bartulem/usv-playpen/blob/main/LICENSE)
[![repo size](https://img.shields.io/github/repo-size/bartulem/usv-playpen)](https://github.com/bartulem/usv-playpen/)
[![downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fzenodo.org%2Fapi%2Frecords%2F7926971&query=%24.stats.downloads&label=downloads&color=blue)](https://doi.org/10.5281/zenodo.7926971)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![last commit](https://img.shields.io/github/last-commit/bartulem/usv-playpen)](https://github.com/bartulem/usv-playpen/commits/main)
[![Contributors](https://img.shields.io/github/contributors/bartulem/usv-playpen)](https://github.com/bartulem/usv-playpen/graphs/contributors)
![shields.io-issues](https://img.shields.io/github/issues/bartulem/usv-playpen)
[![Documentation Status](https://readthedocs.org/projects/usv-playpen/badge/?version=latest)](https://usv-playpen.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/bartulem/usv-playpen/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bartulem/usv-playpen/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bartulem/usv-playpen/graph/badge.svg)](https://codecov.io/gh/bartulem/usv-playpen)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Release](https://img.shields.io/github/v/release/bartulem/usv-playpen)](https://github.com/bartulem/usv-playpen/releases)
[![GitHub stars](https://img.shields.io/github/stars/bartulem/usv-playpen?style=social)](https://github.com/bartulem/usv-playpen/)
[![GitHub forks](https://img.shields.io/github/forks/bartulem/usv-playpen?style=social)](https://github.com/bartulem/usv-playpen/)

# usv-playpen

![](https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/src/usv_playpen/img/usv_playpen_gui.png)

GUI/CLI to facilitate conducting experiments with multi-probe e-phys
(Neuropixels), multichannel audio (Avisoft) and multi-camera video (Loopbio)
acquisition. Developed for behavioral recording purposes at the
[Princeton Neuroscience Institute](https://pni.princeton.edu/) 2021-27
([Falkner](https://www.falknerlab.com/)/[Murthy](https://murthylab.princeton.edu/)
labs). Due to necessary proprietary software, recordings can only be performed
on OS Windows. The data processing, analysis and visualization branches of the
GUI are platform-independent.

## Prerequisites

- [git](https://git-scm.com/download/) (if on Windows, add PATH to USER
  VARIABLES)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (if on Windows,
  add PATH to USER VARIABLES)
- [das](https://janclemenslab.org/das/) (install in standalone conda
  environment)
- [vocalocator-ssl](https://github.com/Aramist/vocalocator-ssl) (install in
  standalone conda environment)

## Installation and updating

Clone the repository and set up virtual environment with _uv_:

```bash
git clone https://github.com/bartulem/usv-playpen.git
cd usv-playpen
uv venv --python=3.13
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

### Updating to a specific version

Navigate to the cloned repository. To see which version you currently have:

```
git describe --tags
```

To see all available versions:

```
git fetch --tags
git tag --sort=-version:refname
```

To update to a specific version (works identically on Linux, macOS, and
Windows):

```
git fetch --tags
git checkout v0.9.5
uv sync --reinstall-package usv-playpen
```

Or, if you have a GPU:

```
git fetch --tags
git checkout v0.9.5
uv sync --reinstall-package usv-playpen --extra gpu
```

Replace `v0.9.5` with whichever version you want. The
`--reinstall-package usv-playpen` flag is required to force uv to recompute the
package version from the new git tag; without it the GUI may display a stale
version number. All machines should be set to the same tag to guarantee
identical behaviour.

### Development installs

The commands above install the runtime dependencies only. To work on the code
itself (tests, notebooks, docs), also install the development dependency groups:

```bash
uv sync --all-groups               # macOS, or Linux without an NVIDIA GPU
uv sync --all-groups --all-extras  # Linux with an NVIDIA GPU (adds jax[cuda12])
```

Do not pass `--all-extras` on macOS: the `gpu` extra has no macOS build and the
sync fails. Then enable the git hooks once per clone:

```bash
pre-commit install                        # formatting checks on commit
pre-commit install --hook-type pre-push   # full test suite before each push
```

and run the test suite with `pytest -n auto`. See
[CONTRIBUTING](.github/CONTRIBUTING.md) for the full development workflow.

### Disk usage

`uv` keeps downloaded wheels and build artifacts in its cache (`uv cache dir`
prints the location; `~/.cache/uv` on Linux). With the torch/jax/CUDA stack the
cache reaches **tens of GB** and grows with every `uv sync`. Where disk space or
a home-directory quota is limited, run **`uv cache clean`** periodically. It
only removes files that can be downloaded again; the next `uv sync` re-fetches
what it needs. If conda is installed on the same machine, `conda clean --all`
clears its package cache as well.

## Usage

Run the GUI with:

```bash
activate-pni && usv-playpen
```

User guide with detailed instructions is available
[here](https://usv-playpen.readthedocs.io/en/latest/).
