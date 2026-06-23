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
            "make_usv_spectrograms_bool": False,
            "make_qlvm_torus_traversal_video_bool": False,
            "make_embedding_thumbnails_bool": False,
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
        'USVSpectrogramPlotter': mocker.patch('usv_playpen.visualizations.visualize_data.USVSpectrogramPlotter'),
        'QLVMTorusTraversalVideo': mocker.patch('usv_playpen.visualizations.visualize_data.QLVMTorusTraversalVideo'),
        'render_embedding_thumbnails_for_cohort': mocker.patch('usv_playpen.visualizations.visualize_data.render_embedding_thumbnails_for_cohort'),
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


def test_make_usv_spectrograms_logic(mock_settings, mock_dependencies):
    """
    The USV spectrogram figure is PER-SESSION: it must run once per root
    directory (inside the loop), independent of the cohort torus block.
    """
    mock_settings['visualize_booleans']['make_usv_spectrograms_bool'] = True
    root_dirs = ['/fake/dir1', '/fake/dir2', '/fake/dir3']

    visualizer = Visualizer(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    visualizer.visualize_data()

    mock_plotter = mock_dependencies['USVSpectrogramPlotter']
    assert mock_plotter.call_count == len(root_dirs)
    assert mock_plotter.return_value.make_usv_spectrograms.call_count == len(root_dirs)
    # the cohort torus block must NOT have run
    assert mock_dependencies['QLVMTorusTraversalVideo'].call_count == 0


def test_make_qlvm_torus_traversal_video_logic(mock_settings, mock_dependencies):
    """
    The QLVM torus video is COHORT-LEVEL: it must run exactly once (outside the
    per-session loop), regardless of how many root directories are given.
    """
    mock_settings['visualize_booleans']['make_qlvm_torus_traversal_video_bool'] = True
    root_dirs = ['/fake/dir1', '/fake/dir2', '/fake/dir3']

    visualizer = Visualizer(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    visualizer.visualize_data()

    mock_video = mock_dependencies['QLVMTorusTraversalVideo']
    assert mock_video.call_count == 1
    mock_video.return_value.make_video.assert_called_once()


def test_make_embedding_thumbnails_logic(mock_settings, mock_dependencies):
    """
    The embedding + per-category thumbnails figure is COHORT-LEVEL: it must run
    exactly once (outside the per-session loop), regardless of how many root
    directories are given.
    """
    mock_settings['visualize_booleans']['make_embedding_thumbnails_bool'] = True
    root_dirs = ['/fake/dir1', '/fake/dir2', '/fake/dir3']

    visualizer = Visualizer(
        input_parameter_dict=mock_settings,
        root_directories=root_dirs
    )
    visualizer.visualize_data()

    mock_thumbnails = mock_dependencies['render_embedding_thumbnails_for_cohort']
    assert mock_thumbnails.call_count == 1


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
    # the USV sequence figure controls (column 3)
    for attr in ("usv_seq_cb", "usv_seq_fig_format_cb", "usv_seq_start_edit",
                 "usv_seq_duration_edit", "usv_seq_embedding_cb",
                 "usv_seq_mask_cb", "usv_seq_raw_cb", "usv_seq_boundaries_cb",
                 "usv_seq_boundary_clustering_cb", "usv_seq_mark_cb"):
        assert hasattr(win, attr), f"Visualize tab missing USV-sequence widget: {attr}"
    # the sequence selectors are seeded from settings
    assert win.usv_seq_embedding in ("qlvm", "vae")
    win.close()


def test_gui_usv_sequence_control_coupling(qtbot, monkeypatch, tmp_path):
    """Boundaries apply to BOTH embeddings (VAE has category boundaries too); the
    clustering selector is enabled only when boundaries = Yes."""
    monkeypatch.chdir(tmp_path)
    win = _make_main_window(qtbot)
    win.visualize_one()

    # VAE + boundaries Yes -> boundaries stay enabled (no longer QLVM-only) and the
    # clustering selector is enabled.
    win.usv_seq_embedding_cb.setCurrentText("vae")
    win.usv_seq_boundaries_cb.setCurrentText("Yes")
    win._update_usv_seq_enabled_state()
    assert win.usv_seq_boundaries_cb.isEnabled()
    assert win.usv_seq_boundary_clustering_cb.isEnabled()
    assert win.usv_seq_boundary_clustering_label.isEnabled()

    # boundaries No -> clustering selector disabled (either embedding)
    win.usv_seq_boundaries_cb.setCurrentText("No")
    win._update_usv_seq_enabled_state()
    assert win.usv_seq_boundaries_cb.isEnabled()
    assert not win.usv_seq_boundary_clustering_cb.isEnabled()
    assert not win.usv_seq_boundary_clustering_label.isEnabled()

    # QLVM + boundaries Yes -> clustering selector enabled
    win.usv_seq_embedding_cb.setCurrentText("qlvm")
    win.usv_seq_boundaries_cb.setCurrentText("Yes")
    win._update_usv_seq_enabled_state()
    assert win.usv_seq_boundary_clustering_cb.isEnabled()
    assert win.usv_seq_boundary_clustering_label.isEnabled()
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
