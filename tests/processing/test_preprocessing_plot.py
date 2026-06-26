"""
@author: bartulem
Test for processing/preprocessing_plot.py SummaryPlotter.

The preprocessing summary is a single large multi-panel figure method that
mixes numpy statistics with matplotlib rendering over a Motif/imgstore video
session. imgstore (`new_for_filename`) and the session-metadata loader are
substituted so the whole figure pipeline — A/V IPI discrepancy histograms +
scatter, the per-mouse subject table (`m_*` device branch), and the phidget
humidity/lux/temperature insets (`s_*` device branch) — runs against a tiny
synthetic session and writes a real summary figure to `<root>/sync/`.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from scipy.io import wavfile

from usv_playpen.processing.preprocessing_plot import SummaryPlotter


def _ipi_block(rng, *, with_nidq=False):
    """
    Description
    -----------
    Build one device's IPI-discrepancy block: the per-pulse A/V start
    discrepancies (ms, with at least one |value| >= 10 to exercise the
    wide-axis branch) and their video start frames; optionally the NIDQ
    inset arrays.

    Parameters
    ----------
    rng (np.random.Generator)
        RNG source.
    with_nidq (bool)
        Whether to attach the NIDQ inset arrays.

    Returns
    -------
    block (dict)
        The device's IPI block.
    """

    discrepancy = np.concatenate([rng.normal(0, 3, 40), [12.0, -11.0]])
    block = {
        "ipi_discrepancy_ms": discrepancy,
        "video_ipi_start_frames": np.arange(discrepancy.size, dtype=float),
    }
    if with_nidq:
        block["nidq_ipi_discrepancy_ms"] = rng.normal(0, 2, 30)
        block["nidq_ipi_start_samples"] = np.arange(30, dtype=float)
    return block


def _make_img_store_mock():
    """
    Description
    -----------
    Construct a MagicMock standing in for an imgstore handle, exposing the
    attributes `preprocessing_summary` reads: `frame_count`,
    `get_frame_metadata()['frame_time']`, `_format`, and the `user_metadata`
    block (motif version / gain / exposure / hardware frame rate).

    Parameters
    ----------

    Returns
    -------
    store (MagicMock)
        Configured imgstore stand-in.
    """

    store = MagicMock()
    store.frame_count = 1000
    store.get_frame_metadata.return_value = {
        "frame_time": np.linspace(0.0, 600.0, 1000)
    }
    store._format = "mjpeg"
    store.user_metadata = {
        "motif_version": "1.0",
        "gain": 1,
        "exposuretime": 5000,
        "hwframerate": 150,
    }
    return store


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_preprocessing_summary_writes_figure(tmp_path, mocker):
    """
    Description
    -----------
    End-to-end `preprocessing_summary` over a synthetic session with two
    devices (`m_usgh` driving the subject-info table + NIDQ inset, `s_avisoft`
    driving the phidget humidity/lux/temperature insets): the method must
    compute the discrepancy statistics, update + save the environment
    metadata, render the 2 x n_devices grid, and write the summary figure
    under `<root>/sync/`.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Session root.
    mocker (pytest_mock.MockerFixture)
        Patches imgstore + metadata loaders.

    Returns
    -------
    None
    """

    root = tmp_path / "20240101_120000"
    (root / "video" / "cam.01_session").mkdir(parents=True)
    (root / "video" / "cam.01_session" / "metadata.yaml").write_text("{}")
    (root / "video" / "sess_camera_frame_count_dict.json").write_text(
        json.dumps({"total_video_time_least": 600.0})
    )
    cropped = root / "audio" / "cropped_to_video"
    cropped.mkdir(parents=True)
    wavfile.write(cropped / "ch01.wav", 250000, np.zeros(1000, dtype=np.int16))

    mocker.patch(
        "usv_playpen.processing.preprocessing_plot.new_for_filename",
        return_value=_make_img_store_mock(),
    )
    metadata = {
        "Subjects": [
            {"subject_id": "M", "genotype_strain": "WT", "sex": "male",
             "dob": "2024-01-01", "weight": "25", "housing": "group"},
            {"subject_id": "F", "genotype_strain": "WT", "sex": "female",
             "dob": "2024-01-02", "weight": "22", "housing": "group"},
        ],
        "Session": {"experimenter": "bm"},
        "Environment": {},
    }
    mocker.patch(
        "usv_playpen.processing.preprocessing_plot.load_session_metadata",
        return_value=(metadata, root / "x_metadata.yaml"),
    )
    save_mock = mocker.patch(
        "usv_playpen.processing.preprocessing_plot.save_session_metadata"
    )

    rng = np.random.default_rng(0)
    ipi_discrepancy_dict = {
        "m_usgh": _ipi_block(rng, with_nidq=True),
        "s_avisoft": _ipi_block(rng),
    }
    phidget_data_dictionary = {
        "humidity": np.concatenate([rng.uniform(40, 60, 50), [np.nan]]),
        "lux": np.concatenate([rng.uniform(0, 5, 50), [np.nan]]),
        "temperature": np.concatenate([rng.uniform(20, 25, 50), [np.nan]]),
    }

    plotter = SummaryPlotter(
        root_directory=str(root),
        message_output=lambda *_a, **_k: None,
    )
    plotter.preprocessing_summary(
        ipi_discrepancy_dict=ipi_discrepancy_dict,
        phidget_data_dictionary=phidget_data_dictionary,
    )

    # The environment medians were written back into the metadata and saved.
    assert "luminance_lux" in metadata["Environment"]
    assert "temperature_celsius" in metadata["Environment"]
    assert "humidity_percent" in metadata["Environment"]
    assert save_mock.call_count == 1

    out_files = list((root / "sync").glob("20240101_120000_summary.*"))
    assert out_files, "summary figure not written under <root>/sync/"
    assert out_files[0].stat().st_size > 5_000


def _img_store_mock_with_user_meta():
    """
    Description
    -----------
    imgstore stand-in whose `user_metadata` carries the legacy comma-joined
    subject / cage / weight / dob / sex / strain / housing / experimenter
    fields the metadata-absent branch parses directly off the Motif store.

    Parameters
    ----------

    Returns
    -------
    store (MagicMock)
        Configured imgstore stand-in.
    """

    store = _make_img_store_mock()
    store.user_metadata = {
        "motif_version": "1.0", "gain": 1, "exposuretime": 5000, "hwframerate": 150,
        "experimenter": "bm",
        "subject": "M,F", "cage": "c1,c2", "weight": "25,22",
        "dob": "2024-01-01,2024-01-02", "sex": "male,female",
        "strain": "WT,WT", "housing": "group,group",
    }
    return store


def _base_session(tmp_path, *, with_wav=True):
    """
    Description
    -----------
    Build the common session skeleton (a Motif-style video subdir, the camera
    frame-count JSON, and optionally a cropped-to-video WAV).

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Session root parent.
    with_wav (bool)
        Whether to write the cropped-to-video WAV.

    Returns
    -------
    root (pathlib.Path)
        The created session root.
    """

    root = tmp_path / "20240101_120000"
    (root / "video" / "cam.01_session").mkdir(parents=True)
    (root / "video" / "cam.01_session" / "metadata.yaml").write_text("{}")
    (root / "video" / "sess_camera_frame_count_dict.json").write_text(
        json.dumps({"total_video_time_least": 600.0})
    )
    cropped = root / "audio" / "cropped_to_video"
    cropped.mkdir(parents=True)
    if with_wav:
        wavfile.write(cropped / "ch01.wav", 250000, np.zeros(1000, dtype=np.int16))
    return root


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_preprocessing_summary_no_metadata_parses_motif_fields(tmp_path, mocker):
    """
    Description
    -----------
    With no session metadata file, `preprocessing_summary` must fall back to
    parsing the subject/cage/etc. fields straight off the Motif `user_metadata`
    (the metadata-absent branch), and a device whose discrepancies are all
    within ±10 ms must take the fixed [-10.5, 10.5] axis branch.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Session root parent.
    mocker (pytest_mock.MockerFixture)
        Patches imgstore + metadata loader.

    Returns
    -------
    None
    """

    root = _base_session(tmp_path)
    mocker.patch(
        "usv_playpen.processing.preprocessing_plot.new_for_filename",
        return_value=_img_store_mock_with_user_meta(),
    )
    mocker.patch(
        "usv_playpen.processing.preprocessing_plot.load_session_metadata",
        return_value=(None, None),
    )

    rng = np.random.default_rng(1)
    small = rng.uniform(-4.0, 4.0, 40)  # all |x| < 10 -> fixed-axis branch
    ipi_discrepancy_dict = {
        "m_usgh": {"ipi_discrepancy_ms": small,
                   "video_ipi_start_frames": np.arange(small.size, dtype=float)},
        "s_avisoft": _ipi_block(rng),
    }
    phidget_data_dictionary = {
        "humidity": rng.uniform(40, 60, 50),
        "lux": rng.uniform(0, 5, 50),
        "temperature": rng.uniform(20, 25, 50),
    }

    SummaryPlotter(
        root_directory=str(root),
        message_output=lambda *_a, **_k: None,
    ).preprocessing_summary(
        ipi_discrepancy_dict=ipi_discrepancy_dict,
        phidget_data_dictionary=phidget_data_dictionary,
    )

    assert list((root / "sync").glob("*_summary.*")), "summary figure not written"


def test_preprocessing_summary_missing_wav_raises(tmp_path, mocker):
    """
    Description
    -----------
    A session with no cropped-to-video WAV must raise FileNotFoundError rather
    than silently producing a summary without the audio channel context.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Session root parent.
    mocker (pytest_mock.MockerFixture)
        Patches the metadata loader (unused before the raise).

    Returns
    -------
    None
    """

    root = _base_session(tmp_path, with_wav=False)
    rng = np.random.default_rng(2)
    ipi = {"m_usgh": _ipi_block(rng)}
    phidget = {"humidity": rng.uniform(40, 60, 10),
               "lux": rng.uniform(0, 5, 10),
               "temperature": rng.uniform(20, 25, 10)}

    plotter = SummaryPlotter(
        root_directory=str(root),
        message_output=lambda *_a, **_k: None,
    )
    with pytest.raises(FileNotFoundError, match="No .wav files"):
        plotter.preprocessing_summary(
            ipi_discrepancy_dict=ipi, phidget_data_dictionary=phidget,
        )
