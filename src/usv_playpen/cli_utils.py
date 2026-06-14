"""
@author: bartulem
Helper functions for command line interfaces.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import click


class StringTuple(click.ParamType):
    """
    A custom click type that accepts a comma-separated pair of strings
    and converts it into a tuple. e.g., "Head, Nose" -> ('Head', 'Nose')
    """

    name = "STRING,STRING"  # This name is shown in help text, e.g., [--option STRING,STRING]

    def convert(self, value, param, ctx):
        """
        Description
        -----------
        Click parameter-type converter. Called by click for each value
        passed to a CLI option using this type. Converts the user's
        comma-separated input into the canonical typed structure that
        the consuming command expects.

        Parameters
        ----------
        value (str)
            Raw input string from the command line.
        param (click.Parameter)
            The click parameter object that owns this conversion
            (used by click's failure machinery).
        ctx (click.Context | None)
            The active click context (also passed through on failure).

        Returns
        -------
        converted
            The successfully converted value. Calls `self.fail(...)` on
            invalid input, which raises a `click.BadParameter` and never
            returns.
        """
        try:
            # Split the input string by the comma
            parts = [part.strip() for part in value.split(",")]

            # Validate that there are exactly two parts
            if len(parts) != 2:
                # This is the standard way to raise a validation error in a custom type
                self.fail(
                    f"'{value}' is not a valid pair. "
                    "Please provide two strings separated by a comma (e.g., 'Head,Nose').",
                    param,
                    ctx,
                )

            # Return the validated and converted value
            return tuple(parts)

        except ValueError:
            self.fail(
                f"'{value}' could not be parsed as a comma-separated pair.", param, ctx
            )


def find_nested_key_paths(d: dict, target_key: str) -> list[str]:
    """
    Description
    -----------
    Returns the dot-separated paths of every location at which ``target_key``
    occurs in a nested dictionary. This is the name-to-path resolver used by
    ``modify_settings_json_for_cli``: a short CLI option names a bare settings
    key, which may occur in more than one settings block. Collecting all paths
    lets the caller detect and warn about that ambiguity, then perform the actual
    write against a single explicit path via ``set_nested_value_by_path``.

    The traversal records a match at the current dictionary level before
    descending, and visits children in insertion order, so the first element of
    the returned list is always the shallowest, earliest match — the path
    ``modify_settings_json_for_cli`` selects when an option is ambiguous.

    Parameters
    ----------
    d (dict)
        The dictionary to search within.
    target_key (str)
        The key to find.

    Returns
    -------
    (list[str])
        Every dot-separated path at which ``target_key`` occurs, in traversal
        order (so index 0 is the one ``modify_settings_json_for_cli`` writes to).
        Empty if the key is absent.
    """

    paths: list[str] = []

    def _walk(sub: dict, prefix: str) -> None:
        # record a match at this level before descending (shallowest/earliest wins)
        if target_key in sub:
            paths.append(f"{prefix}.{target_key}" if prefix else target_key)
        for key, val in sub.items():
            if isinstance(val, dict):
                _walk(val, f"{prefix}.{key}" if prefix else key)

    _walk(d, "")
    return paths


def modify_settings_json_for_cli(
    ctx: click.Context,
    provided_params: list,
    parameters_lists: list | None = None,
    settings_dict: str | None = None,
) -> dict:
    """
    Description
    -----------
    Modifies the `*_settings.json` file to include
    parameters provided via the command line interface.

    A provided parameter that does not match any key in the loaded settings
    dictionary (e.g. a typo'd, renamed or stale CLI option) is reported on
    stderr and skipped, rather than being dropped silently as before.

    A provided parameter whose (short) name matches a key in more than one
    settings block is ambiguous: it is applied to the first match (the block
    ``find_nested_key_paths`` lists first) and the ambiguity is reported on stderr,
    naming every location and the one chosen, so the user can switch to the
    explicit dot-path override form to target a specific block.

    Parameters
    ----------
    ctx (click.Context)
        Click context containing the parameters.
    provided_params (list)
        Parameters provided via the command line interface.
    parameters_lists (list)
        List of parameters that are expected to be lists.
        If a parameter is not in this list, it is treated as a single value.
    settings_dict (str)
        Settings dictionary file (analyses, processing or visualizations).

    Returns
    -------
    settings_parameter_dict (dict)
        The modified settings dictionary with the provided parameters.
    """

    with open(
        pathlib.Path(__file__).parent / f"_parameter_settings/{settings_dict}.json",
        encoding="utf-8",
    ) as input_json_file:
        settings_parameter_dict = json.load(input_json_file)

    if parameters_lists is None:
        parameters_lists = []

    for param_name in provided_params:
        if param_name not in parameters_lists:
            param_value = ctx.params[param_name]
        elif isinstance(ctx.params[param_name], tuple):
            param_value = list(ctx.params[param_name])
        else:
            param_value = [ctx.params[param_name]]

        matching_paths = find_nested_key_paths(settings_parameter_dict, param_name)
        if not matching_paths:
            print(
                f"Warning: CLI option '{param_name}' did not match any key in "
                f"'{settings_dict}.json'; its value was not applied.",
                file=sys.stderr,
            )
            continue

        if len(matching_paths) > 1:
            print(
                f"Warning: CLI option '{param_name}' is ambiguous in "
                f"'{settings_dict}.json' — it occurs at {matching_paths}. "
                f"Applying it to '{matching_paths[0]}'; use the explicit "
                f"dot-path override to target a specific block.",
                file=sys.stderr,
            )

        set_nested_value_by_path(
            settings_parameter_dict, matching_paths[0], param_value
        )

    return settings_parameter_dict


def set_nested_value_by_path(d: dict, path: str, value: Any) -> None:
    """
    Description
    -----------
    Sets a value in a nested dictionary using a dot-separated path. The path
    is validated against the existing structure of ``d`` — if any intermediate
    key does not exist, is not itself a dict, or the leaf key is missing, a
    KeyError is raised that names the offending sub-path. This is important
    when the dictionary represents a typed configuration schema: a typo in a
    CLI override (e.g. 'video.general.delete-post-copy' with a hyphen) would
    otherwise silently create a new, useless key rather than updating the
    real one.

    Parameters
    ----------
    d (dict)
        The dictionary to modify. Must contain the full target path already.
    path (str)
        The dot-separated path to the key where the value should be set.
        Empty strings and components starting/ending with a dot are rejected.
    value (Any)
        The value to set at the specified path.

    Returns
    -------
    (None)
        This function modifies the dictionary in place and does not return anything.

    Raises
    ------
    ValueError
        If ``path`` is empty, contains empty components, or is not a string.
    KeyError
        If any component of the path does not correspond to an existing key
        in the dictionary, or an intermediate key exists but is not itself
        a dictionary.
    """

    if not isinstance(path, str) or path == "":
        raise ValueError("set_nested_value_by_path: 'path' must be a non-empty string.")

    keys = path.split(".")
    if any(k == "" for k in keys):
        raise ValueError(
            f"set_nested_value_by_path: 'path' has empty component(s): {path!r}."
        )

    current_level = d
    traversed = []
    for key in keys[:-1]:
        if not isinstance(current_level, dict):
            raise KeyError(
                f"set_nested_value_by_path: '{'.'.join(traversed)}' is not a dict "
                f"in the target dictionary (full path: {path!r})."
            )
        if key not in current_level:
            raise KeyError(
                f"set_nested_value_by_path: unknown key "
                f"'{'.'.join(traversed + [key])}' (full path: {path!r})."
            )
        current_level = current_level[key]
        traversed.append(key)

    if not isinstance(current_level, dict):
        raise KeyError(
            f"set_nested_value_by_path: '{'.'.join(traversed)}' is not a dict "
            f"in the target dictionary (full path: {path!r})."
        )
    if keys[-1] not in current_level:
        raise KeyError(
            f"set_nested_value_by_path: unknown key "
            f"'{'.'.join(traversed + [keys[-1]])}' (full path: {path!r})."
        )
    current_level[keys[-1]] = value


def _convert_value(s: str) -> Any:
    """
    Description
    -----------
    Converts a string to its appropriate type based on its content.

    Parameters
    ----------
    s (str)
        The string to convert. It can be a boolean, integer, float, or string.
        - If the string is 'true' or 'false', it will be converted to a boolean.
        - If the string can be converted to an integer or float, it will be converted accordingly.
        - Otherwise, it will return the string with any surrounding quotes removed.

    Returns
    -------
    (Any)
        The converted value:
        - `True` or `False` for boolean strings.
        - An `int` or `float` for numeric strings.
        - A `str` with surrounding quotes removed for other strings.
    """

    s = s.strip()  # Remove leading/trailing whitespace
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        f = float(s)
        if f.is_integer():
            return int(f)
        return f
    except ValueError:
        return s.strip("\"'")


def override_toml_values(overrides: list, exp_settings_dict: dict) -> dict:
    """
    Description
    -----------
    Overrides values in a settings dictionary based on a list of override strings.

    Parameters
    ----------
    overrides (list)
        A list of strings in the format "key.path=value" where:
        - `key.path` is the dot-separated path to the value to be set.
    exp_settings_dict (dict)
        The dictionary to modify with the overrides.
        The keys are dot-separated paths to the values to be set.

    Returns
    -------
    exp_settings_dict (dict)
        The modified settings dictionary with the provided overrides applied.
    """

    for override_str in overrides:
        if "=" not in override_str:
            continue

        key_path, value_str = override_str.split("=", 1)
        final_value: Any

        if "," in value_str:
            items = value_str.split(",")
            final_value = [_convert_value(item) for item in items]
        else:
            final_value = _convert_value(value_str)

        set_nested_value_by_path(exp_settings_dict, key_path, final_value)

    return exp_settings_dict
