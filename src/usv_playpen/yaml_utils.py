"""
@author: bartulem
Utility functions for modifying the metadata YAML file.
"""

from __future__ import annotations

import numbers
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

# Matches strings that YAML 1.1 would silently coerce to a non-string type.
# Covers: integers (with optional sign/underscores), ISO dates (YYYY-MM-DD),
# ISO datetimes (YYYY-MM-DDTHH:MM:SS…), and YAML 1.1 boolean literals.
_YAML11_INT_PATTERN = re.compile(r'^[+-]?[0-9][0-9_]*$')
_YAML11_DATE_PATTERN = re.compile(
    r'^\d{4}-\d{1,2}-\d{1,2}'       # date part: YYYY-M-D or YYYY-MM-DD
    r'([T ]\d{2}:\d{2}(:\d{2})?.*)?$'  # optional time part
)
_YAML11_BOOL_PATTERN = re.compile(
    r'^(y|Y|yes|Yes|YES|n|N|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$'
)


# Custom Dumper to format lists in flow style (e.g., [1, 2, 3])
# while keeping dictionaries in block style for overall readability.
class SmartDumper(yaml.Dumper):
    """
    Custom yaml.Dumper that emits simple (scalar-only) lists in flow style
    (e.g., [1, 2, 3]) while keeping nested or complex lists and dictionaries
    in block style for readability. Also single-quotes strings that YAML 1.1
    would otherwise silently coerce to ints, dates, or booleans.
    """

    def represent_list(self, data):
        """
        Description
        Represents a Python list in YAML. Chooses flow style (inline) when
        every element is a scalar (str, number, or None); otherwise falls
        back to block style (one item per line).

        Parameters
        data (list)
            The list to serialize.

        Returns
        yaml.nodes.SequenceNode
            A YAML sequence node emitted in flow or block style.
        """

        is_simple_list = all(isinstance(item, (str, numbers.Number)) or item is None for item in data)

        if is_simple_list:
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        else:
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

    def represent_str(self, data):
        """
        Description
        Represents a Python string in YAML. If the string looks like an integer,
        ISO date/datetime, or YAML 1.1 boolean literal (e.g., 'yes', 'off', '2024-01-01'),
        it is emitted with single quotes so consumers do not silently coerce it
        to a non-string type on load.

        Parameters
        data (str)
            The string value to serialize.

        Returns
        yaml.nodes.ScalarNode
            A YAML scalar node, single-quoted when coercion would occur.
        """

        if (
            _YAML11_INT_PATTERN.match(data)
            or _YAML11_DATE_PATTERN.match(data)
            or _YAML11_BOOL_PATTERN.match(data)
        ):
            return self.represent_scalar('tag:yaml.org,2002:str', data, style="'")
        return self.represent_scalar('tag:yaml.org,2002:str', data)

    def represent_numpy_scalar(self, data):
        """
        Description
        Represents a NumPy scalar (np.float64, np.int32, np.bool_, etc.) in
        YAML by first converting it to the equivalent native Python scalar via
        .item(). Without this, PyYAML falls back to the generic Python-object
        representer and emits tagged values like '!!python/object/apply:numpy
        .float64', which round-trip poorly across environments and break
        downstream consumers that expect plain scalars.

        Parameters
        data (numpy.generic)
            A NumPy scalar instance (any subclass of numpy.generic).

        Returns
        yaml.nodes.Node
            A YAML node produced by the Dumper's representer for the
            corresponding native Python type (int, float, bool, or str).
        """

        return self.represent_data(data.item())

SmartDumper.add_representer(list, SmartDumper.represent_list)
SmartDumper.add_representer(str, SmartDumper.represent_str)
# Register a multi-representer so every numpy scalar subclass (np.float64,
# np.int32, np.bool_, np.str_, etc.) is funneled through represent_numpy_scalar
# and emitted as its native Python equivalent.
SmartDumper.add_multi_representer(np.generic, SmartDumper.represent_numpy_scalar)


def load_session_metadata(root_directory: str, logger: Callable = print) -> tuple[dict | None, Path | None]:
    """
    Finds and loads the session-specific metadata.yaml file from a given directory.

    Parameters
    root_directory (str)
        The directory to search for the metadata file.

    Returns
    Tuple containing (loaded_data, file_path), or (None, None) if not found.
    """

    path = Path(root_directory)
    metadata_path_list = list(path.glob('*_metadata.yaml'))
    if not metadata_path_list:
        return None, None

    metadata_path = metadata_path_list[0]
    try:
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f), metadata_path
    except yaml.YAMLError as e:
        logger(f"Error loading metadata file: {e}")
        return None, None


def save_session_metadata(data: dict, filepath: Path, logger: Callable = print) -> None:
    """
    Saves the given data back to the specified metadata file path.

    Parameters
    data (dict)
        The metadata to save.
    filepath (Path)
        The path to the metadata file.

    Returns
    """

    try:
        with open(filepath, 'w') as f:
            yaml.dump(data, f, Dumper=SmartDumper, default_flow_style=False, sort_keys=False, indent=2)
    except yaml.YAMLError as e:
        logger(f"Error saving metadata file: {e}")
