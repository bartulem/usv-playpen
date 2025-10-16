"""
@author: bartulem
Utility functions for modifying the metadata YAML file.
"""

import numbers
import yaml
from pathlib import Path

# Custom Dumper to format lists in flow style (e.g., [1, 2, 3])
# while keeping dictionaries in block style for overall readability.
class SmartDumper(yaml.Dumper):
    def represent_list(self, data):
        is_simple_list = all(isinstance(item, (str, numbers.Number, bool)) or item is None for item in data)

        if is_simple_list:
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        else:
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

SmartDumper.add_representer(list, SmartDumper.represent_list)


def load_session_metadata(root_directory: str, logger: callable = print) -> tuple[dict | None, Path | None]:
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

    try:
        path = Path(root_directory)
        metadata_path_list = list(path.glob('*_metadata.yaml'))
        if not metadata_path_list:
            return None, None

        metadata_path = metadata_path_list[0]
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f), metadata_path
    except Exception as e:
        logger(f"Error loading metadata file: {e}")
        return None, None


def save_session_metadata(data: dict, filepath: Path, logger: callable = print) -> None:
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
    except Exception as e:
        logger(f"Error saving metadata file: {e}")