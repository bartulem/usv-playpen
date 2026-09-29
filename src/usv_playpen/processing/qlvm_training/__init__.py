# ABOUTME: Vendored torch QLVM (QMC latent-variable model) training kernels, from
# ABOUTME: the specgen spec_gen_full_pipeline package, driven by processing.train_qlvm.
"""Vendored QLVM training kernels.

This subpackage is a near-verbatim vendor of the QMC latent-variable model
training core from the ``specgen`` ``spec_gen_full_pipeline`` project
(``training/models/qlvm``): the torch ``QMCLVM`` + basis functions
(``qmc_base``), the fixed quasi-random lattice generators (``sampling``), the
evidence/log-prob losses (``losses``), the train/test epoch loops (``loop``) and
the checkpoint save/load helpers (``checkpoint``). It is kept close to the
original so it tracks that validated implementation; the usv-playpen-native
orchestration (``.npz`` I/O, settings, the CLI, and the decoder-weights
export) lives in ``processing/train_qlvm.py``.

These kernels are inherently ``torch`` (training); the usv-playpen QLVM
*inference* path is the separate torch-free JAX port in ``processing/qlvm_model.py``.
The in-repo inference (``infer-qlvm-latents``) no longer reads the exported
decoder ``state_dict`` ``.npz``: production embeds with QLVM model package
cells. This trainer is kept until its port of the current trainer lands.
"""
from __future__ import annotations
