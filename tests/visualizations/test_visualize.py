"""
@author: bartulem
Test visualizations module.
"""

import pytest
from usv_playpen.visualizations.visualize_data import Visualizer

@pytest.fixture
def mock_settings():
    """Provides a mocked visualizations_settings dictionary for tests."""

    settings = {
        "visualize_booleans": {
            "make_neuronal_tuning_figures_bool": False,
            "make_behavioral_videos_bool": False,
        },
        "credentials_directory": "/fake/credentials",
        "send_email": {
            "send_message": {"receivers": []},
            "visualizations_pc_choice": "Test PC",
            "experimenter": "Tester"
        },
        "neuronal_tuning_figures": {},
        "make_behavioral_videos": {
            "arena_directory": "/fake/arena/dir",
            "speaker_audio_file": "/fake/speaker.wav",
        }
    }
    return settings


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external class dependencies for the Visualizer class."""

    mocked_classes = {
        'NeuronalTuningFigureMaker': mocker.patch('usv_playpen.visualizations.visualize_data.NeuronalTuningFigureMaker'),
        'Create3DVideo': mocker.patch('usv_playpen.visualizations.visualize_data.Create3DVideo'),
        'Messenger': mocker.patch('usv_playpen.visualizations.visualize_data.Messenger'),
    }
    return mocked_classes

def test_visualize_data_no_booleans_true(mock_settings, mock_dependencies):
    """
    Tests that if all boolean flags are False, no visualization methods are called.
    """
    visualizer = Visualizer(
        input_parameter_dict=mock_settings,
        root_directories=['/fake/dir1']
    )
    visualizer.visualize_data()

    # check that only the Messenger was called to send start/end emails
    assert mock_dependencies['Messenger'].return_value.send_message.call_count == 2

    # ensure no other visualization classes were even initialized
    for name, mock_class in mock_dependencies.items():
        if name != 'Messenger':
            assert mock_class.call_count == 0


def test_make_neuronal_tuning_figures_logic(mock_settings, mock_dependencies):
    """
    Tests that `NeuronalTuningFigureMaker.make_neuronal_tuning_figures` is called when the flag is True.
    """

    mock_settings['visualize_booleans']['make_neuronal_tuning_figures_bool'] = True
    root_dirs = ['/fake/dir1', '/fake/dir2']

    visualizer = Visualizer(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    visualizer.visualize_data()

    mock_maker = mock_dependencies['NeuronalTuningFigureMaker']
    assert mock_maker.call_count == len(root_dirs)
    mock_maker.return_value.make_neuronal_tuning_figures.assert_called()
    assert mock_maker.return_value.make_neuronal_tuning_figures.call_count == len(root_dirs)


def test_make_behavioral_videos_logic(mock_settings, mock_dependencies):
    """
    Tests that `Create3DVideo.visualize_in_video` is called when the flag is True.
    """
    mock_settings['visualize_booleans']['make_behavioral_videos_bool'] = True
    root_dirs = ['/fake/dir1']

    visualizer = Visualizer(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    visualizer.visualize_data()

    mock_video_creator = mock_dependencies['Create3DVideo']
    assert mock_video_creator.call_count == len(root_dirs)
    mock_video_creator.return_value.visualize_in_video.assert_called_once()


def test_multiple_tasks_are_called(mock_settings, mock_dependencies):
    """
    Tests that when multiple boolean flags are True, all corresponding tasks are executed.
    """
    mock_settings['visualize_booleans']['make_neuronal_tuning_figures_bool'] = True
    mock_settings['visualize_booleans']['make_behavioral_videos_bool'] = True
    root_dirs = ['/fake/dir1', '/fake/dir2']

    visualizer = Visualizer(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    visualizer.visualize_data()

    mock_maker = mock_dependencies['NeuronalTuningFigureMaker']
    assert mock_maker.call_count == len(root_dirs)
    assert mock_maker.return_value.make_neuronal_tuning_figures.call_count == len(root_dirs)

    # check Create3DVideo calls
    mock_video_creator = mock_dependencies['Create3DVideo']
    assert mock_video_creator.call_count == len(root_dirs)
    assert mock_video_creator.return_value.visualize_in_video.call_count == len(root_dirs)


# ---------------------------------------------------------------------------
# GUI smoke tests
#
# These exist primarily to catch settings-key / dict-path regressions: the
# GUI reads several settings on construction, and a wrong-tab read (the
# pre-existing `smoothing_sd` bug we just fixed) silently raised KeyError on
# load. A construct-and-tear-down test would have caught it.
#
# We use pytest-qt's `qtbot` fixture so a QApplication is in place.
# ---------------------------------------------------------------------------


def test_gui_main_window_constructs_without_error(qtbot, monkeypatch, tmp_path):
    """The GUI main window can be instantiated. Catches KeyError /
    AttributeError on construction (e.g. wrong settings dict path)."""
    # Avoid blocking dialogs / file system writes during import-time hooks.
    monkeypatch.chdir(tmp_path)
    from usv_playpen.usv_playpen_gui import initialize_main_window
    _app, win = initialize_main_window(no_splash=True)
    qtbot.addWidget(win)
    assert win is not None
    win.close()


def test_gui_settings_dicts_have_expected_top_level_keys(qtbot, monkeypatch, tmp_path):
    """The GUI loads settings JSONs on construction. Verify that the
    block names the GUI reads from are still present after schema renames."""
    monkeypatch.chdir(tmp_path)
    from usv_playpen.usv_playpen_gui import initialize_main_window
    _app, win = initialize_main_window(no_splash=True)
    qtbot.addWidget(win)
    # The triage-related settings live under analyses_input_dict:
    assert "calculate_neuronal_tuning_curves" in win.analyses_input_dict
    assert "detect_interesting_tuning_neurons" in win.analyses_input_dict
    # And the figure-format / cmap dropdowns read from `figures` in
    # visualizations_input_dict:
    assert "figures" in win.visualizations_input_dict
    fig_block = win.visualizations_input_dict["figures"]
    assert "fig_format" in fig_block and "cmap" in fig_block
    win.close()


def _make_main_window(qtbot):
    """Construct the main GUI window and register it with qtbot."""
    from usv_playpen.usv_playpen_gui import initialize_main_window
    _app, win = initialize_main_window(no_splash=True)
    qtbot.addWidget(win)
    return win


# Note: Record and Process tabs require state set by the splash / exp-id
# selection flow (e.g. self.sleap_inference_dir_global) that a direct
# smoke test bypasses; testing them needs full-flow integration. Analyze
# and Visualize only need the JSON dicts loaded at GUI init, so they're
# straightforward to smoke.


def test_gui_analyze_tab_constructs(qtbot, monkeypatch, tmp_path):
    """The Analyze tab can be initialized without crashing.
    Catches dict-path / settings-key regressions for the analyses settings
    block (e.g. the smoothing_sd-on-wrong-tab bug from earlier)."""
    monkeypatch.chdir(tmp_path)
    win = _make_main_window(qtbot)
    win.analyze_one()
    assert hasattr(win, "AnalysesSettings")
    # the new triage-related entries must be visible on this tab as
    # the tab build directly reads them at construction time:
    assert hasattr(win, "include_partner_vocalization_tuning_cb")
    assert hasattr(win, "behavioral_min_occupancy_seconds")
    assert hasattr(win, "usv_property_min_occupancy_seconds")
    assert hasattr(win, "smoothing_sd")
    win.close()


def test_gui_visualize_tab_constructs(qtbot, monkeypatch, tmp_path):
    """The Visualize tab can be initialized without crashing.
    Catches issues in the new figure-format / colormap dropdowns."""
    monkeypatch.chdir(tmp_path)
    win = _make_main_window(qtbot)
    win.visualize_one()
    assert hasattr(win, "VisualizationsSettings")
    # the default figure-format / colormap dropdowns
    assert hasattr(win, "default_fig_format_cb")
    assert hasattr(win, "default_cmap_cb")
    win.close()


def test_gui_returning_to_main_window_doesnt_crash(qtbot, monkeypatch, tmp_path):
    """Round-trip: open a tab, return to main, open another."""
    monkeypatch.chdir(tmp_path)
    win = _make_main_window(qtbot)
    win.analyze_one()
    win.main_window()
    win.visualize_one()
    win.main_window()


# ---------------------------------------------------------------------------
# Record / Process tab smoke tests
#
# These tabs require globals normally populated by the splash + exp-id
# selection flow (sleap_inference_dir_global, avisoft_*_global, etc.).
# We invoke the helper that sets them, then call the tab constructor.
# The result catches schema regressions in the much larger Record /
# Process surfaces (~3500 lines combined) without booting the full GUI
# flow.
# ---------------------------------------------------------------------------


def test_gui_record_tab_constructs(qtbot, monkeypatch, tmp_path):
    """The Record tab can be initialized after the exp-id state has been
    primed via _save_variables_based_on_exp_id. Catches regressions in
    avisoft / coolterm / credentials path wiring."""
    monkeypatch.chdir(tmp_path)
    win = _make_main_window(qtbot)
    win._save_variables_based_on_exp_id()
    win.record_one()
    assert hasattr(win, "Record")
    # A handful of widgets that record_one creates and that downstream
    # signal-handlers reference. If any go missing we'll get AttributeError.
    for attr in ("recorder_settings_edit", "avisoft_base_edit",
                 "avisoft_config_edit", "coolterm_base_edit"):
        assert hasattr(win, attr), f"Record tab missing widget: {attr}"
    win.close()


def test_gui_process_tab_constructs(qtbot, monkeypatch, tmp_path):
    """The Process tab can be initialized after exp-id state has been primed.
    Catches schema regressions in the synchronize/anipose/das wiring."""
    monkeypatch.chdir(tmp_path)
    win = _make_main_window(qtbot)
    win._save_variables_based_on_exp_id()
    win.process_one()
    assert hasattr(win, "ProcessSettings")
    win.close()


def test_gui_record_then_process_then_main(qtbot, monkeypatch, tmp_path):
    """Round-trip: Record → main → Process → main. No state leak between
    transitions."""
    monkeypatch.chdir(tmp_path)
    win = _make_main_window(qtbot)
    win._save_variables_based_on_exp_id()
    win.record_one()
    win.main_window()
    win._save_variables_based_on_exp_id()
    win.process_one()
    win.main_window()
    win.close()
    win.close()
