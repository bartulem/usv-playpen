# ABOUTME: Top-level package for trainable / advanced USV box detectors that propose
# ABOUTME: one bounding box per ultrasonic vocalization for SAM2 box-prompted masking.
"""box_detectors — trainable USV box detectors for SAM2 box-prompting.

A self-contained subproject providing learned alternatives to the unlearned
connected-component box detector used in Step 2 of ``spec_gen_full_pipeline``.
Each detector proposes one bounding box per USV; those boxes are fed to
``SAM2ImagePredictor.predict(box=...)``.

See ``README.md`` for the box-format contract, env/install, and run commands;
``INTEGRATION.md`` for wiring ``--detector {cc,unet,yolo}`` into the existing
boxprompt path (note only — the existing pipeline is never edited).
"""

__all__ = ["common"]
