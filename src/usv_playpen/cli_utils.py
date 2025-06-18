"""
@author: bartulem
Helper functions for command line interfaces.
"""

import click
import json
import pathlib
from typing import Any


class StringTuple(click.ParamType):
    """
    A custom click type that accepts a comma-separated pair of strings
    and converts it into a tuple. e.g., "Head, Nose" -> ('Head', 'Nose')
    """
    name = "STRING,STRING"  # This name is shown in help text, e.g., [--option STRING,STRING]

    def convert(self, value, param, ctx):
        """
        This method performs the conversion. It's called by click for each
        value provided to the option.
        """
        try:
            # Split the input string by the comma
            parts = [part.strip() for part in value.split(',')]

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
            self.fail(f"'{value}' could not be parsed as a comma-separated pair.", param, ctx)


def set_nested_key(d: dict = None,
                   target_key: str = None,
                   value: Any = None) -> bool:
    """
    Description
    -----------
    Recursively finds a key in a nested dictionary and sets its value.
    -----------

    Parameters
    ----------
    d (dict)
        The dictionary to search within.
    target_key (str)
        The key to find.
    value (typing.Any)
        The new value to set for the target_key.
    ----------

    Returns
    ----------
    (bool)
        True if the key was found and updated, False otherwise.
    ----------
    """
    # change key if it exists at the current level of the dictionary
    if target_key in d:
        d[target_key] = value
        return True

    # keep searching nested dictionaries.
    for key, val in d.items():
        if isinstance(val, dict):
            if set_nested_key(val, target_key, value):
                return True

    return False

def modify_settings_json_for_cli(ctx: click.Context = None,
                                 provided_params: list = None,
                                 parameters_lists: list = None,
                                 settings_dict: str = None) -> dict:
    """
    Description
    -----------
    Modifies the `*_settings.json` file to include
    parameters provided via the command line interface.
    -----------

    Parameters
    ----------
    ctx (click.Context)
        Click context containing the parameters.
    parameters_lists (list)
        List of parameters that are expected to be lists.
        If a parameter is not in this list, it is treated as a single value.
    provided_params (list)
        Parameters provided via the command line interface.
    settings_dict (str)
        Settings dictionary file (analyses, processing or visualizations).
    ----------

    Returns
    ----------
    settings_parameter_dict (dict)
        The modified settings dictionary with the provided parameters.
    ----------
    """

    with open((pathlib.Path(__file__).parent / f'_parameter_settings/{settings_dict}.json'), 'r') as input_json_file:
        settings_parameter_dict = json.load(input_json_file)

    if parameters_lists is None:
        parameters_lists = []

    for param_name in provided_params:
        if param_name not in parameters_lists:
            param_value = ctx.params[param_name]
        else:
            if isinstance(ctx.params[param_name], tuple):
                param_value = list(ctx.params[param_name])
            else:
                param_value = [ctx.params[param_name]]

        set_nested_key(settings_parameter_dict, param_name, param_value)

    return settings_parameter_dict

def set_nested_value_by_path(d: dict, path: str, value: Any) -> None:
    """
    Description
    -----------
    Sets a value in a nested dictionary using a dot-separated path.
    -----------

    Parameters
    ----------
    d (dict)
        The dictionary to modify.
    path (str)
        The dot-separated path to the key where the value should be set.
    value (Any)
        The value to set at the specified path.
    ----------

    Returns
    ----------
    (None)
        This function modifies the dictionary in place and does not return anything.
    ----------
    """

    keys = path.split('.')
    current_level = d
    for key in keys[:-1]:
        current_level = current_level.setdefault(key, {})
    current_level[keys[-1]] = value

def _convert_value(s: str) -> Any:
    """
    Description
    -----------
    Converts a string to its appropriate type based on its content.
    -----------

    Parameters
    ----------
    s (str)
        The string to convert. It can be a boolean, integer, float, or string.
        - If the string is 'true' or 'false', it will be converted to a boolean.
        - If the string can be converted to an integer or float, it will be converted accordingly.
        - Otherwise, it will return the string with any surrounding quotes removed.
    ----------

    Returns
    ----------
    (Any)
        The converted value:
        - `True` or `False` for boolean strings.
        - An `int` or `float` for numeric strings.
        - A `str` with surrounding quotes removed for other strings.
    ----------
    """

    s = s.strip() # Remove leading/trailing whitespace
    if s.lower() == 'true':
        return True
    if s.lower() == 'false':
        return False
    try:
        f = float(s)
        if f.is_integer():
            return int(f)
        return f
    except ValueError:
        return s.strip('"\'')

def override_toml_values(overrides: list, exp_settings_dict: dict) -> dict:
    """
    Description
    -----------
    Overrides values in a settings dictionary based on a list of override strings.
    -----------

    Parameters
    ----------
    overrides (list)
        A list of strings in the format "key.path=value" where:
        - `key.path` is the dot-separated path to the value to be set.
    exp_settings_dict (dict)
        The dictionary to modify with the overrides.
        The keys are dot-separated paths to the values to be set.
    ----------

    Returns
    ----------
    exp_settings_dict (dict)
        The modified settings dictionary with the provided overrides applied.
    ----------
    """

    for override_str in overrides:
        if '=' not in override_str:
            continue

        key_path, value_str = override_str.split('=', 1)
        final_value: Any

        if ',' in value_str:
            items = value_str.split(',')
            final_value = [_convert_value(item) for item in items]
        else:
            final_value = _convert_value(value_str)

        set_nested_value_by_path(exp_settings_dict, key_path, final_value)

    return exp_settings_dict