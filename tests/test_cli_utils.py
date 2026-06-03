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
    modify_settings_json_for_cli,
    override_toml_values,
    set_nested_key,
    set_nested_value_by_path,
)


# set_nested_key

def test_set_nested_key_top_level():
    d = {"a": 1, "b": 2}
    assert set_nested_key(d, "a", 99) is True
    assert d["a"] == 99


def test_set_nested_key_nested():
    d = {"outer": {"inner": {"target": 0}}}
    assert set_nested_key(d, "target", 5) is True
    assert d["outer"]["inner"]["target"] == 5


def test_set_nested_key_not_found_returns_false():
    d = {"a": {"b": 1}}
    assert set_nested_key(d, "missing", 5) is False
    assert d == {"a": {"b": 1}}  # untouched


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
