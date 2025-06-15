"""
@author: bartulem
Creates/saves text file with list of videos to run SLEAP inference on.
"""

import json
import os
import platform

class PrepareClusterJob:
    def __init__(self, input_parameter_dict: dict = None,
                 root_directory: list[str] = None,
                 message_output: callable = None) -> None:

        """
        Initializes the PrepareClusterJob class.

        Parameter
        ---------
        root_directory (list of str)
            Root directories for data; defaults to None.
        input_parameter_dict (dict)
            Processing parameters; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        -------
        """

        if input_parameter_dict is None:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_parameter_settings/processing_settings.json'), 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)['prepare_cluster_job']
        else:
            self.input_parameter_dict = input_parameter_dict['prepare_cluster_job']

        if root_directory is None:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_parameter_settings/processing_settings.json'), 'r') as json_file:
                self.root_directory = json.load(json_file)['prepare_cluster_job']['root_directory']
        else:
            self.root_directory = root_directory

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

    def video_list_to_txt(self) -> None:
        """
        Description
        ----------
        This method creates a text file (job_list.txt) with
        a list of videos to run SLEAP inference on.

        NB: You need this file w/ to run SLEAP inference on the cluster!
        ----------

        Parameter
        ---------
        camera_names (list)
            Cameras used for recording video.
        inference_root_dir (str)
            Root directory of inference slurm files.
        centroid_model_path (str)
            Path to the centroid model.
        centered_instance_model_path (str)
            Path to the centered instance model.

        Returns
        -------
        job_list (.txt)
            List of sessions to run inference on.
        """

        if platform.system() == 'Windows':
            spock_converted_first_model_path = self.input_parameter_dict['centroid_model_path'].replace('\\', '/').replace('F:', '/mnt/cup/labs/falkner')
            if self.input_parameter_dict['centered_instance_model_path'] == '':
                spock_converted_second_model_path = ''
            else:
                spock_converted_second_model_path = self.input_parameter_dict['centered_instance_model_path'].replace('\\', '/').replace('F:', '/mnt/cup/labs/falkner')
        elif platform.system() == 'Linux':
            spock_converted_first_model_path = self.input_parameter_dict['centroid_model_path'].replace('mnt', 'mnt/cup/labs')
            if self.input_parameter_dict['centered_instance_model_path'] == '':
                spock_converted_second_model_path = ''
            else:
                spock_converted_second_model_path = self.input_parameter_dict['centered_instance_model_path'].replace('mnt', 'mnt/cup/labs')
        else:
            spock_converted_first_model_path = self.input_parameter_dict['centroid_model_path'].replace('Volumes', 'mnt/cup/labs')
            if self.input_parameter_dict['centered_instance_model_path'] == '':
                spock_converted_second_model_path = ''
            else:
                spock_converted_second_model_path = self.input_parameter_dict['centered_instance_model_path'].replace('Volumes', 'mnt/cup/labs')

        with open(f"{self.input_parameter_dict['inference_root_dir']}{os.sep}job_list.txt", mode='w') as job_list_file:
            for root_dir in self.root_directory:

                if platform.system() == 'Windows':
                    spock_converted_root_dir = root_dir.replace('\\', '/').replace('F:', '/mnt/cup/labs/falkner')
                elif platform.system() == 'Linux':
                    spock_converted_root_dir = root_dir.replace('mnt', 'mnt/cup/labs')
                else:
                    spock_converted_root_dir = root_dir.replace('Volumes', 'mnt/cup/labs')

                for video_processed_dir in os.listdir(f"{root_dir}{os.sep}video"):
                    if video_processed_dir.isnumeric():
                        for video_dir in os.listdir(f"{root_dir}{os.sep}video{os.sep}{video_processed_dir}"):
                            if os.path.isdir(f"{root_dir}{os.sep}video{os.sep}{video_processed_dir}{os.sep}{video_dir}") and video_dir in self.input_parameter_dict['camera_names']:
                                for video_dir_item in os.listdir(f"{root_dir}{os.sep}video{os.sep}{video_processed_dir}{os.sep}{video_dir}"):
                                    if video_dir_item.endswith('.mp4'):
                                        video_path = f"{spock_converted_root_dir}/video/{video_processed_dir}/{video_dir}/{video_dir_item}"
                                        slp_result_path = f"{spock_converted_root_dir}/video/{video_processed_dir}/{video_dir}/{video_dir_item[:-4]}.slp"
                                        if spock_converted_second_model_path == '':
                                            job_list_file.write(f"{spock_converted_first_model_path} {video_path} {slp_result_path}\n")
                                        else:
                                            job_list_file.write(f"{spock_converted_first_model_path} {spock_converted_second_model_path} {video_path} {slp_result_path}\n")
