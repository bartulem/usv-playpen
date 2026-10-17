import pytest

import threading
from importlib import metadata
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
from PyQt6.QtCore import Qt  # Added import
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QPushButton

from usv_playpen import os_utils, usv_playpen_gui
from usv_playpen.usv_playpen_gui import (
    ChemoDialog, EphysDialog, LesionDialog, OptoDialog, _safe_literal_eval,
    replace_name_in_path,
)


def test_replace_name_in_path_pairs_each_destination_with_its_own_name():
    """Each destination is rewritten with the experimenter name that occurs in IT
    (not a cross-product mispairing), names are matched literally (no regex), and a
    destination with no matching name is left unchanged instead of crashing."""
    # experimenter_list order is the REVERSE of the destination order: the old
    # cross-product zip would pair the first path with 'bob' and crash on
    # re.search(...).span() (None); the per-path lookup substitutes correctly.
    assert replace_name_in_path(
        experimenter_list=["bob", "alice"],
        recording_files_destinations=["/data/alice/sess", "/backup/bob/sess"],
        exp_id="EXP1",
    ) == "/data/EXP1/sess,/backup/EXP1/sess"

    # Multi-destination where only one path carries a name: the name-less one is
    # left untouched (the old code crashed on the mispaired None.span()).
    assert replace_name_in_path(
        experimenter_list=["alice"],
        recording_files_destinations=["/data/alice/sess", "/backup/none/sess"],
        exp_id="EXP2",
    ) == "/data/EXP2/sess,/backup/none/sess"

    # A name containing a regex metacharacter is matched literally (not as a regex).
    assert replace_name_in_path(
        experimenter_list=["a.b"],
        recording_files_destinations=["/data/a.b/sess"],
        exp_id="EXP3",
    ) == "/data/EXP3/sess"

@pytest.fixture
def app(qtbot):
    app_obj, window = usv_playpen_gui.initialize_main_window(no_splash=True)
    qtbot.addWidget(window)
    return window


@pytest.fixture
def preserve_settings_toml():
    """Back up and restore the package behavioral_experiments_settings.toml.

    A GUI 'save' exercised under test writes that file via toml.dump, which
    strips comments and reformats it -- churning the tracked package config in
    the repo. Restoring it after the test keeps the working tree clean.
    """
    settings_path = Path(usv_playpen_gui.__file__).parent / '_config' / 'behavioral_experiments_settings.toml'
    original = settings_path.read_text()
    yield
    settings_path.write_text(original)


@pytest.fixture
def preserve_all_settings():
    """
    Description
    -----------
    Back up and restore every package settings file the GUI's label-save
    handlers may rewrite while navigating between windows (the
    ``behavioral_experiments_settings.toml`` plus the three
    ``_parameter_settings`` JSONs), so window-navigation tests never churn the
    tracked configs in the working tree.

    Parameters
    ----------

    Returns
    -------
    None
        Yields control to the test, then restores the original file contents.
    """

    pkg = Path(usv_playpen_gui.__file__).parent
    paths = [
        pkg / '_config' / 'behavioral_experiments_settings.toml',
        pkg / '_parameter_settings' / 'analyses_settings.json',
        pkg / '_parameter_settings' / 'processing_settings.json',
        pkg / '_parameter_settings' / 'visualizations_settings.json',
    ]
    originals = {p: p.read_text() for p in paths}
    yield
    for p, text in originals.items():
        p.write_text(text)

def test_main_window_initialization(app):
    expected_version = metadata.version('usv-playpen').split('.dev')[0].split('.post')[0]
    assert app.windowTitle() == f"USV Playpen v{expected_version}"
    assert app.exp_id_cb.currentText() == "Bartul"

def test_navigation_to_record_window(app, qtbot):
    qtbot.mouseClick(app.button_map['Record'], Qt.MouseButton.LeftButton)
    assert "Record > Select config" in app.windowTitle()
    assert app.recorder_settings_edit is not None

def test_record_one_settings_are_saved(app, qtbot, preserve_settings_toml):
    qtbot.mouseClick(app.button_map['Record'], Qt.MouseButton.LeftButton)

    new_duration = "25.5"

    # Ensure the line edit is cleared (either approach works)
    # Approach 1 (simplest):
    app.video_session_duration.clear()

    # Alternatively (robust): select all then delete
    # qtbot.mouseClick(app.video_session_duration, Qt.MouseButton.LeftButton)
    # qtbot.keyClick(app.video_session_duration, Qt.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
    # qtbot.keyClick(app.video_session_duration, Qt.Key_Delete)

    qtbot.keyClicks(app.video_session_duration, new_duration)

    assert app.exp_settings_dict['video_session_duration'] != 25.5
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert app.exp_settings_dict['video_session_duration'] == 25.5


def test_bundled_metadata_yaml_is_a_dict():
    """
    Description
    -----------
    Guards against silent corruption / accidental truncation of the
    bundled ``_config/_metadata.yaml``. If the file is empty (or anything
    other than a top-level mapping with the ``Session`` / ``Equipment``
    blocks), ``yaml.safe_load`` returns ``None`` and the GUI later
    crashes when ``record_three`` does ``self.metadata_settings['Session']
    ['institution']``. We caught exactly this regression after commit
    ff8eef1 shipped an empty file in v0.10.1 / v0.10.2.

    Parameters
    ----------

    Returns
    -------
    None
    """

    metadata_yaml_path = (
        Path(usv_playpen_gui.__file__).parent / '_config' / '_metadata.yaml'
    )
    with open(metadata_yaml_path, 'r') as metadata_file:
        loaded = yaml.safe_load(metadata_file)
    assert isinstance(loaded, dict), (
        f"{metadata_yaml_path} did not parse to a dict (got {type(loaded).__name__}); "
        "the bundled metadata template must not be empty."
    )
    assert 'Session' in loaded and isinstance(loaded['Session'], dict)
    assert 'Equipment' in loaded and isinstance(loaded['Equipment'], dict)
    for required_session_key in (
        'institution', 'lab', 'session_experiment_code',
        'calibration_session', 'session_usv_playback_file',
        'session_description', 'keywords', 'notes',
    ):
        assert required_session_key in loaded['Session'], (
            f"Session.{required_session_key} missing from bundled metadata YAML"
        )


def test_navigation_to_record_metadata_window(app, qtbot):
    """
    Description
    -----------
    Click ``Record`` → ``Next`` → ``Next`` to walk all the way to the
    ``record_three`` (Metadata) window and assert that the Session
    metadata widgets were constructed from the bundled
    ``_config/_metadata.yaml``. Had this test existed before commit
    ff8eef1, it would have flagged the empty-YAML regression that
    crashed the GUI on the recording PC with
    ``TypeError: 'NoneType' object is not subscriptable``.

    Parameters
    ----------

    Returns
    -------
    None
    """

    qtbot.mouseClick(app.button_map['Record'], Qt.MouseButton.LeftButton)
    assert "Record > Select config" in app.windowTitle()

    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Record > Audio and Video Settings" in app.windowTitle()

    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Record > Metadata" in app.windowTitle()

    assert isinstance(app.metadata_settings, dict)
    assert hasattr(app, 'institution_edit')
    assert app.institution_edit.text() == \
        app.metadata_settings['Session']['institution']
    assert hasattr(app, 'lab_edit')
    assert app.lab_edit.text() == app.metadata_settings['Session']['lab']


# def test_start_recording_button_calls_backend(app, qtbot):
#     qtbot.mouseClick(app.button_map['Record'], Qt.MouseButton.LeftButton)
#     qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
#     qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
#     qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
#     record_button = app.button_map['Record']
#     qtbot.mouseClick(record_button, Qt.MouseButton.LeftButton)
#
# def test_start_processing_button_calls_backend(app, qtbot):
#     qtbot.mouseClick(app.button_map['Process'], Qt.MouseButton.LeftButton)
#     qtbot.keyClicks(app.processing_dir_edit, "/my/test/directory")
#     qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
#     process_button = app.button_map['Process']
#     qtbot.mouseClick(process_button, Qt.MouseButton.LeftButton)


def test_navigation_to_record_conduct_window(app, qtbot, preserve_all_settings):
    """
    Description
    -----------
    Click ``Record`` then ``Next`` three times to walk the full recording
    window chain (``record_one`` -> ``record_two`` -> ``record_three`` ->
    ``record_four``), asserting each window's title in turn and that the final
    'Conduct recording' window (and its widget tree) is built end-to-end.

    Parameters
    ----------
    app (USVPlaypenWindow)
        The main GUI window fixture.
    qtbot (pytestqt.qtbot.QtBot)
        Qt test driver used to synthesise the button clicks.
    preserve_all_settings (None)
        Fixture backing up / restoring the package config files the
        label-save handlers rewrite during navigation.

    Returns
    -------
    None
    """

    qtbot.mouseClick(app.button_map['Record'], Qt.MouseButton.LeftButton)
    assert "Record > Select config" in app.windowTitle()
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Record > Audio and Video Settings" in app.windowTitle()
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Record > Metadata" in app.windowTitle()
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Conduct recording" in app.windowTitle()


def test_navigation_to_process_windows(app, qtbot, preserve_all_settings):
    """
    Description
    -----------
    Clicking ``Process`` must build the processing-settings window
    (``process_one``), and the subsequent ``Next`` must build the 'Conduct
    Processing' window (``process_two``); both window titles are asserted.

    Parameters
    ----------
    app (USVPlaypenWindow)
        The main GUI window fixture.
    qtbot (pytestqt.qtbot.QtBot)
        Qt test driver used to synthesise the button clicks.
    preserve_all_settings (None)
        Fixture backing up / restoring the package config files.

    Returns
    -------
    None
    """

    qtbot.mouseClick(app.button_map['Process'], Qt.MouseButton.LeftButton)
    assert "Process recordings > Settings" in app.windowTitle()
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Conduct Processing" in app.windowTitle()


def test_navigation_to_analyze_windows(app, qtbot, preserve_all_settings):
    """
    Description
    -----------
    Clicking ``Analyze`` must build the analysis-settings window
    (``analyze_one``), and the subsequent ``Next`` must build the 'Conduct
    Analyses' window (``analyze_two``); both window titles are asserted.

    Parameters
    ----------
    app (USVPlaypenWindow)
        The main GUI window fixture.
    qtbot (pytestqt.qtbot.QtBot)
        Qt test driver used to synthesise the button clicks.
    preserve_all_settings (None)
        Fixture backing up / restoring the package config files.

    Returns
    -------
    None
    """

    qtbot.mouseClick(app.button_map['Analyze'], Qt.MouseButton.LeftButton)
    assert "Analyze data > Settings" in app.windowTitle()
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Conduct Analyses" in app.windowTitle()


def test_navigation_to_visualize_windows(app, qtbot, preserve_all_settings):
    """
    Description
    -----------
    Clicking ``Visualize`` must build the visualization-settings window
    (``visualize_one``), and the subsequent ``Next`` must build the 'Conduct
    Visualizations' window (``visualize_two``); both window titles are
    asserted.

    Parameters
    ----------
    app (USVPlaypenWindow)
        The main GUI window fixture.
    qtbot (pytestqt.qtbot.QtBot)
        Qt test driver used to synthesise the button clicks.
    preserve_all_settings (None)
        Fixture backing up / restoring the package config files.

    Returns
    -------
    None
    """

    qtbot.mouseClick(app.button_map['Visualize'], Qt.MouseButton.LeftButton)
    assert "Visualize data > Settings" in app.windowTitle()
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    assert "Conduct Visualizations" in app.windowTitle()


def test_navigation_to_credentials_window(app, qtbot):
    """
    Description
    -----------
    Clicking the login button must open the credentials window
    (``credentials_window``), building its widget tree (asserted via the title
    and the presence of the ``email_address`` field).

    Parameters
    ----------
    app (USVPlaypenWindow)
        The main GUI window fixture.
    qtbot (pytestqt.qtbot.QtBot)
        Qt test driver used to synthesise the button click.

    Returns
    -------
    None
    """

    qtbot.mouseClick(app.login_button, Qt.MouseButton.LeftButton)
    assert "Set credentials" in app.windowTitle()
    assert hasattr(app, 'email_address')


_INTERVENTION_DIALOGS = [
    (ChemoDialog, "chemogenetics"),
    (EphysDialog, "electrophysiology"),
    (LesionDialog, "lesion"),
    (OptoDialog, "optogenetics"),
]


@pytest.fixture
def app_with_subject(app):
    """
    Description
    -----------
    Main window primed with a single metadata subject and the YAML / repository
    persistence side-effects stubbed out, so the intervention dialogs can be
    saved / deleted without touching disk.

    Parameters
    ----------
    app (USVPlaypenWindow)
        The main GUI window fixture.

    Returns
    -------
    app (USVPlaypenWindow)
        The same window with ``metadata_settings`` populated and the
        ``_save_metadata_to_yaml`` / ``_update_subject_in_repository`` writers
        replaced by no-ops.
    """

    app.metadata_settings = {"Subjects": [{"subject_id": "M", "interventions": {}}]}
    app._save_metadata_to_yaml = lambda: None
    app._update_subject_in_repository = lambda s: None
    return app


@pytest.mark.parametrize("dialog_cls,key", _INTERVENTION_DIALOGS)
def test_intervention_dialog_add_mode_saves(app_with_subject, qtbot, dialog_cls, key):
    """
    Description
    -----------
    Each intervention dialog must build its add-mode form, collect the mixed
    combo / line-edit field values on OK, and write them into the chosen
    subject's ``interventions`` block under the dialog's own key.

    Parameters
    ----------
    app_with_subject (USVPlaypenWindow)
        Main window primed with one subject and stubbed persistence.
    qtbot (pytestqt.qtbot.QtBot)
        Qt test driver (registers the dialog for cleanup).
    dialog_cls (type)
        The intervention-dialog class under test.
    key (str)
        The interventions-dict key the dialog writes under.

    Returns
    -------
    None
    """

    dlg = dialog_cls(parent=app_with_subject, subject=None)
    qtbot.addWidget(dlg)
    assert "Add" in dlg.windowTitle()
    dlg.subject_combo.setCurrentText("M")
    dlg.save_and_accept()
    interventions = app_with_subject.metadata_settings["Subjects"][0]["interventions"]
    assert key in interventions
    assert isinstance(interventions[key], dict)


@pytest.mark.parametrize("dialog_cls,key", _INTERVENTION_DIALOGS)
def test_intervention_dialog_edit_mode_populates_and_deletes(
    app_with_subject, qtbot, dialog_cls, key,
):
    """
    Description
    -----------
    In edit mode each dialog must lock the subject selector, pre-populate the
    form from the existing intervention, and (on confirmed delete) remove the
    intervention from the subject.

    Parameters
    ----------
    app_with_subject (USVPlaypenWindow)
        Main window primed with one subject and stubbed persistence.
    qtbot (pytestqt.qtbot.QtBot)
        Qt test driver (registers the dialog for cleanup).
    dialog_cls (type)
        The intervention-dialog class under test.
    key (str)
        The interventions-dict key the dialog edits / deletes.

    Returns
    -------
    None
    """

    subject = {"subject_id": "M", "interventions": {key: {"name": "excitatory"}}}
    app_with_subject.metadata_settings["Subjects"][0] = subject
    dlg = dialog_cls(parent=app_with_subject, subject=subject)
    qtbot.addWidget(dlg)
    assert "Edit" in dlg.windowTitle()
    assert not dlg.subject_combo.isEnabled()

    with patch(
        "usv_playpen.usv_playpen_gui.QMessageBox.question",
        return_value=usv_playpen_gui.QMessageBox.StandardButton.Yes,
    ):
        dlg.delete_intervention()
    assert key not in subject["interventions"]


def test_start_handlers_invoke_backends(app):
    """
    Description
    -----------
    The thin ``_start_*`` handlers must forward to their backend worker objects
    (processing / calibration / recording), and ``_start_recording`` must
    persist any updated metadata returned by the recording backend (and skip
    the save when none is returned).

    Parameters
    ----------
    app (USVPlaypenWindow)
        The main GUI window fixture.

    Returns
    -------
    None
    """

    app.run_processing = MagicMock()
    app._start_processing()
    app.run_processing.prepare_data_for_analyses.assert_called_once()

    app.run_exp = MagicMock()
    app._start_calibration()
    app.run_exp.conduct_tracking_calibration.assert_called_once()

    # Recording that returns no metadata -> no save.
    app.run_exp.conduct_behavioral_recording = MagicMock(return_value=None)
    app._write_metadata_to_disk = MagicMock()
    app._start_recording()
    app._write_metadata_to_disk.assert_not_called()

    # Recording that returns updated metadata -> persisted DIRECTLY, not via the
    # Metadata-gated _save_metadata_to_yaml (which would return without writing
    # while the central widget is ConductRecording, silently losing the metadata).
    app.run_exp.conduct_behavioral_recording = MagicMock(return_value={"Session": {}})
    app._start_recording()
    assert app.metadata_settings == {"Session": {}}
    app._write_metadata_to_disk.assert_called_once()


def test_enable_disable_buttons_toggle_state(app):
    """
    Description
    -----------
    The per-category enable / disable helpers must flip the ``Previous`` /
    ``Main`` / category button enabled-states on their shared ``button_map``.

    Parameters
    ----------
    app (USVPlaypenWindow)
        The main GUI window fixture (supplies the running QApplication).

    Returns
    -------
    None
    """

    keys = ["Previous", "Main", "Visualize", "Analyze", "Process"]
    app.button_map = {k: QPushButton() for k in keys}

    app._disable_visualize_buttons()
    assert not app.button_map["Previous"].isEnabled()
    app._enable_visualize_buttons()
    assert app.button_map["Previous"].isEnabled()
    assert not app.button_map["Visualize"].isEnabled()

    app._disable_analyze_buttons()
    assert not app.button_map["Main"].isEnabled()
    app._enable_analyze_buttons()
    assert app.button_map["Main"].isEnabled()

    app._disable_process_buttons()
    assert not app.button_map["Process"].isEnabled()
    app._enable_process_buttons()
    assert app.button_map["Previous"].isEnabled()


def test_update_subject_in_repository_adds_then_updates(app):
    """
    Description
    -----------
    ``_update_subject_in_repository`` must append a new subject (as an
    independent deep copy), update an existing one in place by ``subject_id``,
    and no-op on a record with no id — persisting via the (stubbed) repository
    writer each time.

    Parameters
    ----------
    app (USVPlaypenWindow)
        The main GUI window fixture.

    Returns
    -------
    None
    """

    app.subject_repository = []
    app._save_subject_to_repository = lambda: None

    app._update_subject_in_repository({"subject_id": "M", "x": 1})
    assert len(app.subject_repository) == 1
    assert app.subject_repository[0] == {"subject_id": "M", "x": 1}

    app._update_subject_in_repository({"subject_id": "M", "x": 2})
    assert len(app.subject_repository) == 1
    assert app.subject_repository[0]["x"] == 2

    app._update_subject_in_repository({})  # no subject_id -> ignored
    assert len(app.subject_repository) == 1


def test_save_metadata_to_yaml_early_return_off_the_metadata_page(app):
    """
    Description
    -----------
    On the main window (where the central widget is not ``Metadata``),
    ``_save_metadata_to_yaml`` must early-return without touching the bundled
    ``_config/_metadata.yaml``.

    Parameters
    ----------
    app (USVPlaypenWindow)
        The main GUI window fixture.

    Returns
    -------
    None
    """

    meta_path = Path(usv_playpen_gui.__file__).parent / '_config' / '_metadata.yaml'
    before = meta_path.read_text()
    app._save_metadata_to_yaml()
    assert meta_path.read_text() == before, "metadata YAML was written on early-return path"


def test_record_pages_have_their_own_containers_and_only_metadata_saves(qtbot, monkeypatch, tmp_path):
    """
    Description
    -----------
    The second recording page (audio, sync and video settings) is built on a
    ``VideoSettings`` container and the third (session metadata) on its own
    ``Metadata`` container. ``_save_metadata_to_yaml`` reads the metadata form
    fields, so it must write only while the metadata page is shown: after the
    metadata page has been visited (its form attributes still exist on the window)
    and the user moves to the audio/video page, a save must return without
    writing, since the form widgets it would read belong to a page that is gone.

    Parameters
    ----------
    qtbot (pytestqt.qtbot.QtBot)
        Qt test helper that owns the window.
    monkeypatch (pytest.MonkeyPatch)
        Redirects the persistent write to a recorder.
    tmp_path (pathlib.Path)
        Working directory, so no real config file is touched.

    Returns
    -------
    None
    """

    win = _subject_window(qtbot, monkeypatch, tmp_path)
    assert type(win.centralWidget()) is usv_playpen_gui.Metadata

    writes = []
    monkeypatch.setattr(win, '_write_metadata_to_disk', lambda: writes.append(True))
    win.record_two()
    assert type(win.centralWidget()) is usv_playpen_gui.VideoSettings
    assert hasattr(win, 'institution_edit')
    win._save_metadata_to_yaml()
    assert writes == []


@pytest.mark.parametrize("raw, expected", [
    ("5", 5),
    ("3.14", 3.14),
    ("(1, 2)", (1, 2)),
    ("[0, 1]", [0, 1]),
    ("-3", -3),
])
def test_safe_literal_eval_parses_valid_literals(raw, expected):
    """The happy path is identical to ast.literal_eval: well-formed number,
    tuple, and list literals parse to the same Python value."""
    assert _safe_literal_eval(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "abc", "(1,", "1 2"])
def test_safe_literal_eval_raises_clear_value_error(raw):
    """Blank or malformed field input raises a ValueError whose message echoes
    the offending value, instead of the cryptic ValueError/SyntaxError that
    ast.literal_eval raises directly."""
    with pytest.raises(ValueError, match="Could not parse the GUI field value"):
        _safe_literal_eval(raw)


def test_nudge_button_icon_up_marks_icon_text_buttons(qtbot):
    """The macOS icon-centering helper marks icon+text buttons once (idempotently)
    and leaves icon-less / text-less buttons untouched."""
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.GlobalColor.black)

    icon_text = QPushButton(QIcon(pixmap), "Go")
    qtbot.addWidget(icon_text)
    usv_playpen_gui._nudge_button_icon_up(icon_text)
    assert icon_text.property("_iconNudged") is True
    usv_playpen_gui._nudge_button_icon_up(icon_text)  # idempotent
    assert icon_text.property("_iconNudged") is True

    icon_only = QPushButton(QIcon(pixmap), "")
    qtbot.addWidget(icon_only)
    usv_playpen_gui._nudge_button_icon_up(icon_only)
    assert not icon_only.property("_iconNudged")

    text_only = QPushButton("NoIcon")
    qtbot.addWidget(text_only)
    usv_playpen_gui._nudge_button_icon_up(text_only)
    assert not text_only.property("_iconNudged")


# Subject repository ("remembered animals") <-> session-record consistency.
#
# `_add_subject` builds its dict from the form widgets only, so keys with no
# widget -- notably `interventions` -- are absent from it. The repository writer
# must therefore never treat an incoming dict as the complete animal.


def _subject_window(qtbot, monkeypatch, tmp_path):
    """Main window on the record_three page, with every persistent write
    redirected away from the user's real config files."""

    monkeypatch.chdir(tmp_path)
    from usv_playpen.usv_playpen_gui import initialize_main_window

    _app, win = initialize_main_window(no_splash=True)
    qtbot.addWidget(win)
    win.subject_repo_path = tmp_path / 'subject_presets.json'
    monkeypatch.setattr(win, '_write_metadata_to_disk', lambda: None)
    win.record_three()
    return win


_REMEMBERED = {
    'subject_id': '181101_2', 'species': 'mouse', 'genotype_strain': 'C57',
    'sex': 'male', 'housing': 'group', 'dob': '', 'weight': '25',
    'estrous_stage': 'N/A', 'estrous_sample_time': '',
    'interventions': {'optogenetics': {'name': 'RED0'}},
}


def _interventions(subject: dict) -> list:
    return sorted((subject or {}).get('interventions', {}) or {})


def test_add_subject_after_recall_keeps_remembered_interventions(qtbot, monkeypatch, tmp_path):
    """Recalling an animal then pressing "Add Subject" must not erase the
    interventions stored in subject_presets.json. `_add_subject` passes a
    form-only dict (no `interventions` key), so a repository writer that
    replaces wholesale destroys them."""

    import copy
    win = _subject_window(qtbot, monkeypatch, tmp_path)
    win.subject_repository = [copy.deepcopy(_REMEMBERED)]
    win.metadata_settings['Subjects'] = []

    win._on_subject_selected_from_completer('181101_2')
    win.subject_form_widgets['genotype_strain'].setText('C57BL/6J')   # a real edit
    win._add_subject()

    assert _interventions(win.subject_repository[0]) == ['optogenetics']
    assert _interventions(win.metadata_settings['Subjects'][0]) == ['optogenetics']


def test_add_subject_by_typed_id_inherits_remembered_interventions(qtbot, monkeypatch, tmp_path):
    """Typing a remembered animal's id (never touching the completer popup) and
    pressing "Add Subject" must seed the session record from the remembered
    preset, so the recording's metadata carries its interventions."""

    import copy
    from PyQt6.QtWidgets import QComboBox

    win = _subject_window(qtbot, monkeypatch, tmp_path)
    win.subject_repository = [copy.deepcopy(_REMEMBERED)]
    win.metadata_settings['Subjects'] = []

    for key, widget in win.subject_form_widgets.items():
        value = str(_REMEMBERED.get(key, ''))
        widget.setCurrentText(value) if isinstance(widget, QComboBox) else widget.setText(value)
    win._add_subject()

    assert _interventions(win.metadata_settings['Subjects'][0]) == ['optogenetics']
    assert _interventions(win.subject_repository[0]) == ['optogenetics']


def test_update_subject_in_repository_still_persists_intervention_deletion(qtbot, monkeypatch, tmp_path):
    """Merging must not resurrect a deleted intervention: the dialogs pass a
    subject whose `interventions` dict is present with the entry removed, so the
    whole sub-dict is replaced."""

    import copy
    win = _subject_window(qtbot, monkeypatch, tmp_path)
    both = copy.deepcopy(_REMEMBERED)
    both['interventions'] = {'optogenetics': {'name': 'RED0'}, 'lesion': {'name': ''}}
    win.subject_repository = [both]

    after_delete = copy.deepcopy(both)
    del after_delete['interventions']['lesion']
    win._update_subject_in_repository(after_delete)

    assert _interventions(win.subject_repository[0]) == ['optogenetics']


def test_subject_completer_refreshes_after_repository_change(qtbot, monkeypatch, tmp_path):
    """A newly remembered animal appears in the autocomplete without having to
    leave and re-enter the record_three page."""

    win = _subject_window(qtbot, monkeypatch, tmp_path)
    win.subject_repository = []
    win._refresh_subject_completer()

    win._update_subject_in_repository({'subject_id': 'NEW_1', 'species': 'mouse'})

    model = win.subject_completer.model()
    listed = [model.data(model.index(i, 0)) for i in range(model.rowCount())]
    assert 'NEW_1' in listed


def test_available_playback_contexts_lists_each_sex_folder_once(tmp_path, monkeypatch):
    """
    Description
    -----------
    The playback-context dropdown offers only contexts with a built repository.
    Each sex folder of the repository root is listed once (the share can take
    seconds per listing), every context of that sex is matched in memory, a
    missing sex folder counts as empty, and nothing found falls back to every
    context.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Temporary repository root.
    monkeypatch (pytest.MonkeyPatch)
        Points the repository root at ``tmp_path`` and counts the listings.

    Returns
    -------
    None
    """

    (tmp_path / 'male').mkdir()
    (tmp_path / 'female').mkdir()
    (tmp_path / 'male' / 'naturalistic_usv_repository_courtship_20260101.h5').touch()
    (tmp_path / 'male' / 'notes.txt').touch()
    (tmp_path / 'female' / 'naturalistic_usv_repository_lone_20260101.h5').touch()
    monkeypatch.setattr(usv_playpen_gui, 'resolve_data_root', lambda key: tmp_path)
    listed = []
    original_iterdir = Path.iterdir

    def counting_iterdir(self):
        listed.append(self.name)
        return original_iterdir(self)

    monkeypatch.setattr(Path, 'iterdir', counting_iterdir)
    available = usv_playpen_gui.USVPlaypenWindow._available_playback_contexts(None)
    assert available == ['courtship_male', 'lone_female']
    assert sorted(listed) == ['female', 'male', 'mixed']

    for path in (tmp_path / 'male').iterdir():
        path.unlink()
    for path in (tmp_path / 'female').iterdir():
        path.unlink()
    assert usv_playpen_gui.USVPlaypenWindow._available_playback_contexts(None) == list(usv_playpen_gui._PLAYBACK_CONTEXTS)


def test_analyze_window_opens_without_listing_and_narrows_contexts_later(app, qtbot, preserve_all_settings, monkeypatch):
    """
    Description
    -----------
    Building the analyses window must not wait on the playback-repository listing
    (a network share, seconds per listing when the link is busy): the dropdown
    opens with every context and is narrowed to the built ones once the
    background listing returns.

    Parameters
    ----------
    app (USVPlaypenWindow)
        The main GUI window fixture.
    qtbot (pytestqt.qtbot.QtBot)
        Qt test driver.
    preserve_all_settings (None)
        Fixture backing up / restoring the package config files.
    monkeypatch (pytest.MonkeyPatch)
        Replaces the listing with one that blocks until released.

    Returns
    -------
    None
    """

    release = threading.Event()

    def slow_listing(self):
        release.wait(10)
        return ['lone_female']

    monkeypatch.setattr(usv_playpen_gui.USVPlaypenWindow, '_available_playback_contexts', slow_listing)
    qtbot.mouseClick(app.button_map['Analyze'], Qt.MouseButton.LeftButton)
    assert app.playback_context_cb.count() == len(usv_playpen_gui._PLAYBACK_CONTEXTS)
    release.set()
    qtbot.waitUntil(lambda: app.playback_context_cb.count() == 1, timeout=5000)
    assert app.playback_context_cb.itemText(0) == 'lone_female'
    assert app.playback_context == 'lone_female'



def test_late_playback_listing_keeps_the_context_the_user_already_chose(app, qtbot, preserve_all_settings, monkeypatch):
    """
    Description
    -----------
    A context the user picks while the background repository listing is still
    running survives the listing's return when it is among the built contexts:
    the dropdown is narrowed, the user's choice stays shown and stored, and the
    settings default does not replace it.

    Parameters
    ----------
    app (USVPlaypenWindow)
        The main GUI window fixture.
    qtbot (pytestqt.qtbot.QtBot)
        Qt test driver.
    preserve_all_settings (None)
        Fixture backing up / restoring the package config files.
    monkeypatch (pytest.MonkeyPatch)
        Replaces the listing with one that blocks until released.

    Returns
    -------
    None
    """

    release = threading.Event()
    default = app.analyses_input_dict['create_naturalistic_usv_playback_wav']['context_label']
    chosen = next(label for label in usv_playpen_gui._PLAYBACK_CONTEXTS if label != default)

    def slow_listing(self):
        release.wait(10)
        return [default, chosen]

    monkeypatch.setattr(usv_playpen_gui.USVPlaypenWindow, '_available_playback_contexts', slow_listing)
    qtbot.mouseClick(app.button_map['Analyze'], Qt.MouseButton.LeftButton)
    app.playback_context_cb.setCurrentIndex(app.playback_context_list.index(chosen))
    app._combo_box_playback_context(app.playback_context_list.index(chosen), variable_id='playback_context')
    assert app.playback_context == chosen
    release.set()
    qtbot.waitUntil(lambda: app.playback_context_cb.count() == 2, timeout=5000)
    assert app.playback_context == chosen
    assert app.playback_context_cb.currentText() == chosen

def test_processing_runner_gets_a_copy_of_the_settings_it_may_fill(app, qtbot, preserve_all_settings):
    """
    Description
    -----------
    The 'Conduct Processing' window hands the processing runner (``Stylist``) a
    deep copy of the processing settings, not the dict the GUI saves to
    ``processing_settings.json`` and re-keys on every experimenter change. The
    processing steps fill the derived model paths (the production QLVM cells,
    the category bundle, the squeak cell) into their settings in place; on the
    shared dict those paths would be saved and then re-keyed to another
    experimenter's directory. After the runner's settings are derived, the
    GUI's own dict must still hold the empty keys the shipped JSON carries.

    Parameters
    ----------
    app (USVPlaypenWindow)
        The main GUI window fixture.
    qtbot (pytestqt.qtbot.QtBot)
        Qt test driver used to synthesise the button clicks.
    preserve_all_settings (None)
        Fixture backing up / restoring the package config files.

    Returns
    -------
    None
    """

    qtbot.mouseClick(app.button_map['Process'], Qt.MouseButton.LeftButton)
    qtbot.mouseClick(app.button_map['Next'], Qt.MouseButton.LeftButton)
    runner_settings = app.run_processing.input_parameter_dict
    assert runner_settings is not app.processing_input_dict
    os_utils.derive_spectrogram_model_paths(runner_settings)
    assert runner_settings['infer_qlvm_latents']['model_cells']
    assert app.processing_input_dict['infer_qlvm_latents']['model_cells'] == {}
    assert app.processing_input_dict['assign_qlvm_categories']['category_directory'] == ''
    assert app.processing_input_dict['infer_qlvm_squeak_latents']['model_cell_directory'] == ''


def test_available_playback_contexts_raises_on_a_missing_settings_key(monkeypatch):
    """
    Description
    -----------
    Only an unreadable settings file (OSError) falls back to offering every playback
    context. A settings file without the ``naturalistic_usv_repository_dir`` key is a
    configuration bug; the broad ``except Exception`` used to hide it behind the
    every-context fallback, and it must now raise.

    Parameters
    ----------
    monkeypatch (pytest.MonkeyPatch)
        Replaces ``resolve_data_root`` with a raising stub.

    Returns
    -------
    None
    """

    def unreadable(key):
        raise OSError("share offline")

    monkeypatch.setattr(usv_playpen_gui, 'resolve_data_root', unreadable)
    assert usv_playpen_gui.USVPlaypenWindow._available_playback_contexts(None) == list(usv_playpen_gui._PLAYBACK_CONTEXTS)

    def missing_key(key):
        raise KeyError(key)

    monkeypatch.setattr(usv_playpen_gui, 'resolve_data_root', missing_key)
    with pytest.raises(KeyError, match="naturalistic_usv_repository_dir"):
        usv_playpen_gui.USVPlaypenWindow._available_playback_contexts(None)


def test_save_subject_to_repository_is_atomic_and_raises_on_non_io_errors(tmp_path, capsys):
    """
    Description
    -----------
    The subject repository is written to a temp sibling and renamed into place: a dump
    that fails half-way leaves the previous file intact (the direct ``open(path, 'w')``
    truncated it first). An I/O failure is reported and swallowed (the GUI keeps
    running); any other error (here a non-serialisable record, TypeError) used to be
    swallowed by ``except Exception`` and must now raise.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Temporary directory holding the repository file.
    capsys (pytest.CaptureFixture)
        Captures the reported I/O error.

    Returns
    -------
    None
    """

    repo_path = tmp_path / 'subjects.json'
    repo_path.write_text('[{"subject_id": "OLD"}]')
    holder = MagicMock()
    holder.subject_repo_path = repo_path
    holder.subject_repository = [{"subject_id": "NEW", "bad": object()}]
    with pytest.raises(TypeError):
        usv_playpen_gui.USVPlaypenWindow._save_subject_to_repository(holder)
    assert repo_path.read_text() == '[{"subject_id": "OLD"}]'
    assert [p.name for p in tmp_path.iterdir()] == ['subjects.json']

    holder.subject_repo_path = tmp_path / 'missing_dir_is_a_file'
    holder.subject_repo_path.write_text('x')
    holder.subject_repo_path = holder.subject_repo_path / 'subjects.json'
    holder.subject_repository = [{"subject_id": "NEW"}]
    usv_playpen_gui.USVPlaypenWindow._save_subject_to_repository(holder)
    assert "Error saving subject repository" in capsys.readouterr().out
