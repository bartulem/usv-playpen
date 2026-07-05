"""
Unit tests for usv_playpen.cli_utils' pure helpers, which test_cli.py exercises
only through mocks: the nested-dict setters, the CLI value coercion, the TOML
override parser, the StringTuple click type, and the new "unmatched CLI option"
warning emitted by modify_settings_json_for_cli.
"""

from types import SimpleNamespace

import click
import pytest

from usv_playpen.cli_utils import (
    StringTuple,
    _convert_value,
    find_nested_key_paths,
    modify_settings_json_for_cli,
    override_toml_values,
    set_nested_value_by_path,
)


# find_nested_key_paths

def test_find_nested_key_paths_single_match():
    d = {"outer": {"inner": {"target": 0}}}
    assert find_nested_key_paths(d, "target") == ["outer.inner.target"]


def test_find_nested_key_paths_absent():
    assert find_nested_key_paths({"a": {"b": 1}}, "missing") == []


def test_find_nested_key_paths_lists_all_in_write_order():
    """Both blocks carry the same leaf key; index 0 must be the path that
    modify_settings_json_for_cli writes to via set_nested_value_by_path (so the
    ambiguity warning names the right chosen block)."""
    d = {"a": {"target": 1}, "b": {"target": 2}}
    paths = find_nested_key_paths(d, "target")
    assert paths == ["a.target", "b.target"]
    set_nested_value_by_path(d, paths[0], 99)
    assert d["a"]["target"] == 99 and d["b"]["target"] == 2  # first match won


def test_find_nested_key_paths_top_level_wins_over_deeper():
    """A current-level key is recorded before descending into siblings, so
    paths[0] is the shallow one and that is what the write targets."""
    d = {"x": {"k": 1}, "k": 5}
    paths = find_nested_key_paths(d, "k")
    assert paths == ["k", "x.k"]
    set_nested_value_by_path(d, paths[0], 7)
    assert d["k"] == 7 and d["x"]["k"] == 1


# set_nested_value_by_path

def test_set_nested_value_by_path_happy():
    d = {"a": {"b": {"c": 1}}}
    set_nested_value_by_path(d, "a.b.c", 99)
    assert d["a"]["b"]["c"] == 99


@pytest.mark.parametrize("path", ["", "a..b"])
def test_set_nested_value_by_path_bad_path_raises_valueerror(path):
    with pytest.raises(ValueError):
        set_nested_value_by_path({"a": {}}, path, 1)


def test_set_nested_value_by_path_non_string_path_raises():
    with pytest.raises(ValueError):
        set_nested_value_by_path({}, 123, 1)


def test_set_nested_value_by_path_unknown_key_raises_keyerror():
    with pytest.raises(KeyError):
        set_nested_value_by_path({"a": {}}, "a.b", 1)


def test_set_nested_value_by_path_non_dict_intermediate_raises_keyerror():
    with pytest.raises(KeyError):
        set_nested_value_by_path({"a": 5}, "a.b", 1)


# _convert_value

@pytest.mark.parametrize("s,expected", [
    ("true", True), ("True", True), ("FALSE", False),
    ("1", 1), ("42", 42), ("-3", -3),
    ("1.5", 1.5), ("1.0", 1),          # int-valued float collapses to int
    ("  7 ", 7),                        # surrounding whitespace stripped
    ('"x"', "x"), ("'y'", "y"),         # surrounding quotes stripped
    ("abc", "abc"),
])
def test_convert_value(s, expected):
    result = _convert_value(s)
    assert result == expected
    assert type(result) is type(expected)


# override_toml_values

def test_override_toml_scalar():
    d = {"video": {"general": {"fps": 1}}}
    override_toml_values(["video.general.fps=150"], d)
    assert d["video"]["general"]["fps"] == 150


def test_override_toml_comma_makes_list():
    d = {"a": {"b": None}}
    override_toml_values(["a.b=1,2,3"], d)
    assert d["a"]["b"] == [1, 2, 3]


def test_override_toml_malformed_without_equals_is_skipped():
    d = {"a": 1}
    override_toml_values(["no_equals_here"], d)  # must not raise
    assert d == {"a": 1}


# StringTuple

def test_string_tuple_valid_pair_trims_whitespace():
    assert StringTuple().convert("Head, Nose", None, None) == ("Head", "Nose")


@pytest.mark.parametrize("value", ["OnlyOne", "a,b,c"])
def test_string_tuple_wrong_arity_raises(value):
    with pytest.raises(click.exceptions.BadParameter):
        StringTuple().convert(value, None, None)


# modify_settings_json_for_cli — unmatched option now warns (was silent)

def test_modify_settings_warns_on_unmatched_param(capsys):
    bogus = "definitely_not_a_real_settings_key_xyz"
    ctx = SimpleNamespace(params={bogus: 5})
    result = modify_settings_json_for_cli(
        ctx, provided_params=[bogus], settings_dict="processing_settings",
    )
    err = capsys.readouterr().err
    assert bogus in err
    assert "did not match" in err
    assert isinstance(result, dict)


def test_modify_settings_warns_on_ambiguous_param(capsys):
    """playback_seed exists in both playback blocks of analyses_settings.json,
    so a short --set is ambiguous: it must warn (naming both locations) and
    apply the value to the first match only, leaving the second untouched."""
    ctx = SimpleNamespace(params={"playback_seed": 7})
    result = modify_settings_json_for_cli(
        ctx, provided_params=["playback_seed"], settings_dict="analyses_settings",
    )
    err = capsys.readouterr().err
    assert "ambiguous" in err
    assert "create_naturalistic_usv_playback_wav.playback_seed" in err
    assert "create_usv_playback_wav.playback_seed" in err
    # Applied to the first match (naturalistic) only; the other block is untouched.
    assert result["create_naturalistic_usv_playback_wav"]["playback_seed"] == 7
    assert result["create_usv_playback_wav"]["playback_seed"] != 7


# modify_settings_json_for_cli — block= scopes an override to a single settings block

def test_modify_settings_block_scopes_shared_key():
    """``batch_size`` occurs in generate_masks, train_qlvm AND train_masks. With
    ``block='train_qlvm'`` the override must land in THAT block only. The pre-block
    resolver wrote a shared key to the first global match (generate_masks), so a
    command's own ``--batch-size`` silently missed its target; this guards the fix."""
    ctx = SimpleNamespace(params={"batch_size": 999})
    result = modify_settings_json_for_cli(
        ctx,
        provided_params=["batch_size"],
        settings_dict="processing_settings",
        block="train_qlvm",
    )
    assert result["train_qlvm"]["batch_size"] == 999
    assert result["generate_masks"]["batch_size"] != 999
    assert result["train_masks"]["batch_size"] != 999


def test_modify_settings_block_emits_no_ambiguity_warning(capsys):
    """A shared key is no longer ambiguous once resolution is scoped to one block,
    so the ambiguity warning the global path prints must not appear."""
    ctx = SimpleNamespace(params={"batch_size": 8})
    modify_settings_json_for_cli(
        ctx,
        provided_params=["batch_size"],
        settings_dict="processing_settings",
        block="train_qlvm",
    )
    assert "ambiguous" not in capsys.readouterr().err


def test_modify_settings_block_warns_when_key_absent_from_block(capsys):
    """A provided param that is not a key of the named block is skipped with a
    block-specific 'did not match' warning, instead of leaking into another block."""
    ctx = SimpleNamespace(params={"batch_size": 8})
    modify_settings_json_for_cli(
        ctx,
        provided_params=["batch_size"],
        settings_dict="processing_settings",
        block="compute_usv_acoustic_features",  # this block has no batch_size key
    )
    err = capsys.readouterr().err
    assert "did not match" in err
    assert "batch_size" in err
    assert "compute_usv_acoustic_features" in err


def test_modify_settings_block_none_falls_back_to_global_first_match():
    """``block=None`` (the default, used by the non-pipeline commands) preserves the
    legacy behavior: a shared key is written to the first block
    ``find_nested_key_paths`` lists (generate_masks for batch_size)."""
    ctx = SimpleNamespace(params={"batch_size": 999})
    result = modify_settings_json_for_cli(
        ctx,
        provided_params=["batch_size"],
        settings_dict="processing_settings",
    )
    assert result["generate_masks"]["batch_size"] == 999
