"""
@author: bartulem
Creates/saves text file with list of videos to run SLEAP inference on.

Local lab-share paths are converted to cluster mount paths (via
``to_cluster_path``/``configure_path``) so that the resulting ``job_list.txt``
references cluster-side locations and is directly runnable on the cluster.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable

from ..os_utils import configure_path, to_cluster_path


class PrepareClusterJob:
    def __init__(self, input_parameter_dict: dict = None,
                 root_directory: list[str] = None,
                 message_output: Callable | None = None) -> None:

        """
        Description
        -----------
        Initializes the PrepareClusterJob class.

        Parameters
        ----------
        input_parameter_dict (dict)
            Processing parameters; defaults to None.
        root_directory (list of str)
            Root directories for data; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        None
        """

        # Load the on-disk settings block only when no explicit input_parameter_dict
        # was supplied; it provides the default for input_parameter_dict alone. The
        # root_directory is always supplied by callers and has no settings default.
        if input_parameter_dict is None:
            with open(pathlib.Path(__file__).parent.parent / '_parameter_settings/processing_settings.json') as json_file:
                _settings = json.load(json_file)['prepare_cluster_job']

        self.input_parameter_dict = input_parameter_dict['prepare_cluster_job'] if input_parameter_dict is not None else _settings
        self.root_directory = root_directory
        self.message_output = message_output if message_output is not None else print

    def video_list_to_txt(self) -> None:
        """
        Description
        -----------
        This method creates a text file (job_list.txt) with
        a list of videos to run SLEAP inference on.

        NB: You need the output text file to run SLEAP inference on the cluster!

        Parameters
        ----------
        None
            Inputs are read from ``self.input_parameter_dict`` (the
            ``prepare_cluster_job`` settings block: ``camera_names``,
            ``inference_root_dir``, ``centroid_model_path``,
            ``centered_instance_model_path``).

        Returns
        -------
        None
            Writes ``job_list.txt`` (the list of sessions to run SLEAP
            inference on) into the inference root directory.
        """

        # SLEAP top-down inference uses a two-stage model pair: the "first" model is the
        # centroid model and the "second" model is the centered-instance model. An empty
        # centered_instance_model_path signals a single-model job (only the first model is
        # written to each job line); otherwise both model paths are written.
        spock_converted_first_model_path = to_cluster_path(self.input_parameter_dict['centroid_model_path'])
        if self.input_parameter_dict['centered_instance_model_path'] == '':
            spock_converted_second_model_path = ''
        else:
            spock_converted_second_model_path = to_cluster_path(self.input_parameter_dict['centered_instance_model_path'])

        pathlib.Path(configure_path(self.input_parameter_dict['inference_root_dir'])).mkdir(parents=True, exist_ok=True)

        with open(pathlib.Path(configure_path(self.input_parameter_dict['inference_root_dir'])) / 'job_list.txt', mode='w') as job_list_file:
            for root_dir in self.root_directory:

                spock_converted_root_dir = to_cluster_path(root_dir)

                # Guard against roots that lack a 'video' subdirectory (e.g. a mistyped
                # or partially-populated session); skipping avoids a FileNotFoundError
                # from iterdir() that would truncate the already-opened job_list.txt and
                # abort all remaining roots in the batch.
                video_root = pathlib.Path(root_dir) / 'video'
                if not video_root.is_dir():
                    self.message_output(f"Skipping {root_dir}: no 'video' subdirectory found.")
                    continue

                for video_processed_dir in video_root.iterdir():
                    if video_processed_dir.name.isnumeric():
                        for video_dir in video_processed_dir.iterdir():
                            if video_dir.is_dir() and video_dir.name in self.input_parameter_dict['camera_names']:
                                for video_dir_item in video_dir.iterdir():
                                    if video_dir_item.name.endswith('.mp4'):
                                        video_path = f"{spock_converted_root_dir}/video/{video_processed_dir.name}/{video_dir.name}/{video_dir_item.name}"
                                        slp_result_path = f"{spock_converted_root_dir}/video/{video_processed_dir.name}/{video_dir.name}/{video_dir_item.stem}.slp"
                                        if spock_converted_second_model_path == '':
                                            job_list_file.write(f"{spock_converted_first_model_path} {video_path} {slp_result_path}\n")
                                        else:
                                            job_list_file.write(f"{spock_converted_first_model_path} {spock_converted_second_model_path} {video_path} {slp_result_path}\n")
