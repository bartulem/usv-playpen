"""
@author: bartulem
Utility functions for modifying the metadata YAML file.
"""

from __future__ import annotations

import numbers
import re
from collections.abc import Callable
from pathlib import Path

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
    def represent_list(self, data):
        is_simple_list = all(isinstance(item, (str, numbers.Number)) or item is None for item in data)

        if is_simple_list:
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        else:
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

    def represent_str(self, data):
        if (
            _YAML11_INT_PATTERN.match(data)
            or _YAML11_DATE_PATTERN.match(data)
            or _YAML11_BOOL_PATTERN.match(data)
        ):
            return self.represent_scalar('tag:yaml.org,2002:str', data, style="'")
        return self.represent_scalar('tag:yaml.org,2002:str', data)

SmartDumper.add_representer(list, SmartDumper.represent_list)
SmartDumper.add_representer(str, SmartDumper.represent_str)


def load_session_metadata(root_directory: str, logger: Callable = print) -> tuple[dict | None, Path | None]:
    """
    Finds and loads the session-specific metadata.yaml file from a given directory.

    Parameters
    ----------
    root_directory (str)
        The directory to search for the metadata file.
    ----------

    Returns
    -------
    Tuple containing (loaded_data, file_path), or (None, None) if not found.
    -------
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
    ----------
    data (dict)
        The metadata to save.
    filepath (Path)
        The path to the metadata file.
    ----------

    Returns
    -------
    -------
    """

    try:
        with open(filepath, 'w') as f:
            yaml.dump(data, f, Dumper=SmartDumper, default_flow_style=False, sort_keys=False, indent=2)
    except yaml.YAMLError as e:
        logger(f"Error saving metadata file: {e}")
