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
        -----------
        Represents a Python list in YAML. Chooses flow style (inline) when
        every element is a scalar (str, number, or None); otherwise falls
        back to block style (one item per line).

        Parameters
        ----------
        data (list)
            The list to serialize.

        Returns
        -------
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
        -----------
        Represents a Python string in YAML. If the string looks like an integer,
        ISO date/datetime, or YAML 1.1 boolean literal (e.g., 'yes', 'off', '2024-01-01'),
        it is emitted with single quotes so consumers do not silently coerce it
        to a non-string type on load.

        Parameters
        ----------
        data (str)
            The string value to serialize.

        Returns
        -------
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
        -----------
        Represents a NumPy scalar (np.float64, np.int32, np.bool_, etc.) in
        YAML by first converting it to the equivalent native Python scalar via
        .item(). Without this, PyYAML falls back to the generic Python-object
        representer and emits tagged values like '!!python/object/apply:numpy
        .float64', which round-trip poorly across environments and break
        downstream consumers that expect plain scalars.

        Parameters
        ----------
        data (numpy.generic)
            A NumPy scalar instance (any subclass of numpy.generic).

        Returns
        -------
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
    Description
    -----------
    Finds and loads the session-specific metadata.yaml file from a given directory.

    Parameters
    ----------
    root_directory (str)
        The directory to search for the metadata file.

    Returns
    -------
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


# Map of recording_codec values stored in behavioral_experiments_settings.toml
# (the choices in the GUI dropdown) to the long-form Motif catalog name
# written into per-session metadata as ``output_file_codec``. The four
# abbreviated entries are Loopbio Motif aliases for h264_nvenc presets:
#   'hq'      -> 'nvenc-slow-yuv420'    (-preset slow)
#   'hq-fast' -> 'nvenc-hq-yuv420'      (-preset hq)
#   'mq'      -> 'nvenc-medium-yuv420'  (-preset medium)
#   'lq'      -> 'nvenc-hp-yuv420'      (-preset hp)
# The three already-long names are passed through unchanged.
_RECORDING_CODEC_LONG_NAME = {
    'hq':                  'nvenc-slow-yuv420',
    'hq-fast':             'nvenc-hq-yuv420',
    'mq':                  'nvenc-medium-yuv420',
    'lq':                  'nvenc-hp-yuv420',
    'nvenc-fast-yuv420_A': 'nvenc-fast-yuv420_A',
    'nvenc-fast-yuv420_B': 'nvenc-fast-yuv420_B',
    'nvenc-ll-yuv420':     'nvenc-ll-yuv420',
}


# Canonical YAML position for each dynamic equipment field. When the field is
# missing from a freshly-toggled equipment block (e.g. one just copied from
# equipment.toml after the catalog had the field stripped), it is inserted
# immediately after the listed sibling key so the on-disk YAML order matches
# the original metadata template instead of having dynamic values trail at
# the end of the block.
_DYNAMIC_FIELD_AFTER_KEY = {
    'sync_LEDs': {
        'device_port': 'device_br',
    },
    'audio_Avisoft': {
        'device_sync': 'device_sn',
        'device_sr':   'device_sync',
    },
    'video_Loopbio': {
        'device_sr':              'device_count',
        'device_sr_calibration':  'device_sr',
        'sensor_count':           'sensor_model',
        'sensor_sn':              'sensor_lens',
        'sensor_exposure_time':   'sensor_sn',
        'sensor_gain':            'sensor_exposure_time',
        'output_file_codec':      'output_file_extension',
    },
}


def _set_block_field_in_position(block: dict, field: str, value, after_key: str) -> bool:
    """
    Description
    -----------
    Set ``block[field] = value`` while preserving canonical YAML key order.

    If ``field`` already exists in ``block``, simple assignment is used --
    Python dict insertion-order means the existing position is preserved.
    If ``field`` is absent, the dict is rebuilt with ``field`` inserted
    immediately after ``after_key`` (the sibling key it should appear next
    to in the on-disk YAML). If ``after_key`` is also absent, ``field`` is
    appended to the end as a best-effort fallback. The function mutates
    ``block`` in place via ``clear()`` + ``update()`` so any external
    references to the block keep pointing at the same dict object.

    Parameters
    ----------
    block (dict)
        Equipment-level dict (e.g. ``Equipment.video_Loopbio``).
    field (str)
        Key being set (e.g. ``'device_sr'``).
    value
        New value to assign.
    after_key (str)
        Sibling key after which ``field`` should appear when it's new.

    Returns
    -------
    changed (bool)
        ``True`` if the block was mutated, ``False`` if the value was
        already present and equal.
    """

    if field in block:
        if block[field] == value:
            return False
        block[field] = value
        return True

    rebuilt: dict = {}
    inserted = False
    for existing_key, existing_value in block.items():
        rebuilt[existing_key] = existing_value
        if existing_key == after_key and not inserted:
            rebuilt[field] = value
            inserted = True
    if not inserted:
        rebuilt[field] = value
    block.clear()
    block.update(rebuilt)
    return True


def sync_equipment_dynamic_fields(metadata_settings: dict, exp_settings_dict: dict) -> bool:
    """
    Description
    -----------
    Reconcile ``metadata_settings['Equipment']`` with the live values that
    live in ``behavioral_experiments_settings.toml``.

    The session metadata YAML carries equipment-level fields whose values
    are not really catalog facts -- they are per-session settings that the
    user picks via the GUI (or ``--set`` overrides). This helper writes
    those settings into the YAML, leaving truly static equipment facts
    untouched. The function never adds equipment blocks the user has not
    selected: if a block is absent from ``metadata_settings['Equipment']``
    (because the user unchecked its equipment checkbox in record_three),
    its fields are skipped.

    Fields synchronised, by block:

      Equipment.sync_LEDs.device_port
          <- exp_settings_dict['arduino_sync_port']
      Equipment.audio_Avisoft.device_sync
          <- exp_settings_dict['audio']['usgh_devices_sync']
      Equipment.audio_Avisoft.device_sr
          <- exp_settings_dict['audio']['devices']['fabtast']
      Equipment.video_Loopbio.device_sr
          <- exp_settings_dict['video']['general']['recording_frame_rate']
      Equipment.video_Loopbio.device_sr_calibration
          <- exp_settings_dict['video']['general']['calibration_frame_rate']
      Equipment.video_Loopbio.output_file_codec
          <- _RECORDING_CODEC_LONG_NAME[
                 exp_settings_dict['video']['general']['recording_codec']
             ]
      Equipment.video_Loopbio.sensor_count
          <- len(exp_settings_dict['video']['general']['expected_cameras'])
      Equipment.video_Loopbio.sensor_sn
          <- sorted ascending list of expected_cameras (matches the YAML's
             original numeric-ascending order)
      Equipment.video_Loopbio.sensor_exposure_time
          <- [exp_settings_dict['video']['cameras_config'][sn]['exposure_time']
              for sn in sorted_expected_cameras]
      Equipment.video_Loopbio.sensor_gain
          <- [exp_settings_dict['video']['cameras_config'][sn]['gain']
              for sn in sorted_expected_cameras]

    Parameters
    ----------
    metadata_settings (dict)
        In-memory session metadata dict (loaded from ``_metadata.yaml``).
    exp_settings_dict (dict)
        In-memory ``behavioral_experiments_settings.toml`` dict.

    Returns
    -------
    changed (bool)
        ``True`` if any field was added or updated, ``False`` if every
        relevant field was already current.
    """

    if not isinstance(metadata_settings, dict):
        return False
    equipment = metadata_settings.get('Equipment')
    if not isinstance(equipment, dict):
        return False

    any_changed = False

    def maybe_set(block_name: str, field: str, new_value) -> None:
        """
        Description
        -----------
        Set `field` to `new_value` inside the named equipment block, in
        the canonical position dictated by `_DYNAMIC_FIELD_AFTER_KEY`,
        and flip the enclosing scope's `any_changed` flag if the value
        actually changed. No-op if `block_name` does not refer to a
        mapping in the equipment dict.

        Parameters
        ----------
        block_name (str)
            Equipment-dict block to write into.
        field (str)
            Field name within the block.
        new_value
            New value; written only if it differs from the current.

        Returns
        -------
        None
        """

        nonlocal any_changed
        block = equipment.get(block_name)
        if not isinstance(block, dict):
            return
        after_key = _DYNAMIC_FIELD_AFTER_KEY.get(block_name, {}).get(field, '')
        if _set_block_field_in_position(block, field, new_value, after_key):
            any_changed = True

    # sync_LEDs.device_port is the canonical Arduino sync port from the toml
    arduino_port = exp_settings_dict.get('arduino_sync_port')
    if isinstance(arduino_port, str):
        maybe_set('sync_LEDs', 'device_port', arduino_port)

    # audio_Avisoft.device_sync <- audio.usgh_devices_sync
    # audio_Avisoft.device_sr   <- audio.devices.fabtast
    audio_section = exp_settings_dict.get('audio') or {}
    if 'usgh_devices_sync' in audio_section:
        maybe_set('audio_Avisoft', 'device_sync', bool(audio_section['usgh_devices_sync']))
    audio_devices = audio_section.get('devices') or {}
    if 'fabtast' in audio_devices:
        maybe_set('audio_Avisoft', 'device_sr', audio_devices['fabtast'])

    # video_Loopbio.* <- video.general.* and per-camera entries in cameras_config
    video_section = exp_settings_dict.get('video') or {}
    video_general = video_section.get('general') or {}
    cameras_config = video_section.get('cameras_config') or {}

    if 'recording_frame_rate' in video_general:
        maybe_set('video_Loopbio', 'device_sr', video_general['recording_frame_rate'])
    if 'calibration_frame_rate' in video_general:
        maybe_set('video_Loopbio', 'device_sr_calibration', video_general['calibration_frame_rate'])
    if 'recording_codec' in video_general:
        codec_short = video_general['recording_codec']
        codec_long = _RECORDING_CODEC_LONG_NAME.get(codec_short, codec_short)
        maybe_set('video_Loopbio', 'output_file_codec', codec_long)
    if 'expected_cameras' in video_general:
        # Sort ascending so sensor_sn matches the original YAML order;
        # exposure / gain lists must follow the same sort order to keep
        # per-camera values aligned with their serial.
        sorted_cams = sorted(str(c) for c in video_general['expected_cameras'])
        sn_values = [int(c) if c.isdigit() else c for c in sorted_cams]
        maybe_set('video_Loopbio', 'sensor_count', len(sorted_cams))
        maybe_set('video_Loopbio', 'sensor_sn', sn_values)

        if cameras_config:
            exposures: list = []
            gains: list = []
            for cam in sorted_cams:
                cam_block = cameras_config.get(cam)
                if not isinstance(cam_block, dict):
                    exposures = []
                    gains = []
                    break
                if 'exposure_time' in cam_block:
                    exposures.append(cam_block['exposure_time'])
                if 'gain' in cam_block:
                    gains.append(cam_block['gain'])
            if exposures and len(exposures) == len(sorted_cams):
                maybe_set('video_Loopbio', 'sensor_exposure_time', exposures)
            if gains and len(gains) == len(sorted_cams):
                maybe_set('video_Loopbio', 'sensor_gain', gains)

    return any_changed


def set_sync_LEDs_device_port(metadata_settings: dict, port: str) -> None:
    """
    Description
    -----------
    Sets ``Equipment.sync_LEDs.device_port`` to the given value, ensuring the
    key is positioned right after ``device_br`` so it lands among the other
    ``device_*`` fields rather than being appended to the end of the
    sync_LEDs block (Python preserves dict insertion order, so naively
    assigning a new key would otherwise put it last).

    The function is a no-op if ``Equipment`` or ``Equipment.sync_LEDs`` is
    missing or not a dict; in that case it neither creates nor reorders
    anything, mirroring the contract of the GUI/record-time call sites that
    only update the port when the sync_LEDs equipment block has explicitly
    been included in the session metadata.

    Parameters
    ----------
    metadata_settings (dict)
        The full in-memory session metadata dictionary (i.e. the dict
        loaded from ``_metadata.yaml`` and later written back to disk).
    port (str)
        New value for ``device_port`` (e.g. ``"COM7"``).

    Returns
    -------
    None
    """

    if not isinstance(metadata_settings, dict):
        return
    equipment = metadata_settings.get('Equipment')
    if not isinstance(equipment, dict):
        return
    sync_block = equipment.get('sync_LEDs')
    if not isinstance(sync_block, dict):
        return

    # remove any existing device_port so we can re-insert it in the
    # canonical position (after device_br) regardless of where it sat
    # before; this guarantees the on-disk YAML order is stable across
    # repeated edits to the Arduino SYNC port field
    if 'device_port' in sync_block:
        del sync_block['device_port']

    rebuilt = {}
    inserted = False
    for existing_key, existing_value in sync_block.items():
        rebuilt[existing_key] = existing_value
        if existing_key == 'device_br' and not inserted:
            rebuilt['device_port'] = port
            inserted = True
    if not inserted:
        # device_br absent — fall back to inserting at the very front so the
        # port still groups with the device_* keys instead of trailing the
        # software / output_file_* metadata
        rebuilt = {'device_port': port, **rebuilt}

    equipment['sync_LEDs'] = rebuilt


def save_session_metadata(data: dict, filepath: Path, logger: Callable = print) -> None:
    """
    Description
    -----------
    Saves the given data back to the specified metadata file path.

    Parameters
    ----------
    data (dict)
        The metadata to save.
    filepath (Path)
        The path to the metadata file.

    Returns
    -------
    None
    """

    try:
        with open(filepath, 'w') as f:
            yaml.dump(data, f, Dumper=SmartDumper, default_flow_style=False, sort_keys=False, indent=2)
    except yaml.YAMLError as e:
        logger(f"Error saving metadata file: {e}")
