"""
@author: bartulem
Creates/saves text file with list of videos to run SLEAP inference on.
"""

import json
import os
import platform

class PrepareClusterJob:
    def __init__(self,
                 root_directory=None,
                 input_parameter_dict=None,
                 message_output=None):

        if input_parameter_dict is None:
            with open('input_parameters.json', 'r') as json_file:
                self.input_parameter_dict = json.load(json_file)['prepare_cluster_job']
        else:
            self.input_parameter_dict = input_parameter_dict['prepare_cluster_job']

        if root_directory is None:
            with open('input_parameters.json', 'r') as json_file:
                self.root_directory = json.load(json_file)['prepare_cluster_job']['root_directory']
        else:
            self.root_directory = root_directory

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

    def video_list_to_txt(self):
        """
        Description
        ----------
        This method creates a text file (job_list.txt) with
        a list of videos to run SLEAP inference on.
        ----------

        Parameter
        ---------
        camera_names: list
            Cameras used for recording video.
        inference_root_dir: str
            Root directory of inference slurm files.
        centroid_model_path: str
            Path to the centroid model.
        centered_instance_model_path: str
            Path to the centered instance model.

        Returns
        -------
        job_list : .txt
            List of sessions to run inference on.
        """

        if platform.system() == 'Windows':
            spock_converted_centroid_model_path = self.input_parameter_dict['centroid_model_path'].replace('\\', '/').replace('F:', '/mnt/cup/labs/falkner')
            spock_converted_centered_instance_model_path = self.input_parameter_dict['centered_instance_model_path'].replace('\\', '/').replace('F:', '/mnt/cup/labs/falkner')
        elif platform.system() == 'Linux':
            spock_converted_centroid_model_path = self.input_parameter_dict['centroid_model_path'].replace('mnt', 'mnt/cup/labs')
            spock_converted_centered_instance_model_path = self.input_parameter_dict['centered_instance_model_path'].replace('mnt', 'mnt/cup/labs')
        else:
            spock_converted_centroid_model_path = self.input_parameter_dict['centroid_model_path'].replace('Volumes', 'mnt/cup/labs')
            spock_converted_centered_instance_model_path = self.input_parameter_dict['centered_instance_model_path'].replace('Volumes', 'mnt/cup/labs')

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
                                        job_list_file.write(f"{spock_converted_centroid_model_path} {spock_converted_centered_instance_model_path} {video_path} {slp_result_path}\n")
