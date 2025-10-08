"""
@author: bartulem
Gets 3D tracked points in metric units, rotated and translated to match arena coordinates.
"""

from __future__ import annotations

import glob
import json
import math
import os
import pathlib
import subprocess
from datetime import datetime

import h5py
import numpy as np
import sleap_anipose
from imgstore import new_for_filename

from .time_utils import *


def find_mouse_names(root_directory: str = None) -> list:
    """
    Description
    ----------
    This function finds the mouse names from the metadata.yaml file.

    NB: Since the metadata.yaml file was not standardized for a while, the function
    check for either the old or the new version of the metadata.yaml file.
    ----------

    Parameters
    ----------
    root_directory (str)
        The directory where of the session.
    ----------

    Returns
    ----------
    track_names (list)
        Mouse names (in the format of "cage_tail-stripe-num").
    ----------
    """

    track_names = []
    for sub_directory in os.listdir(f"{root_directory}{os.sep}video"):
        if (
            os.path.isdir(f"{root_directory}{os.sep}video{os.sep}{sub_directory}")
            and "." in sub_directory
            and "_" in sub_directory
            and "calibration" not in sub_directory
        ):
            img_store = new_for_filename(
                f"{root_directory}{os.sep}video{os.sep}{sub_directory}{os.sep}metadata.yaml"
            )
            user_meta_data = img_store.user_metadata

            if "cage" in user_meta_data.keys() and "subject" in user_meta_data.keys():
                for cage, subject in zip(
                    user_meta_data["cage"].split(","),
                    user_meta_data["subject"].split(","),
                    strict=False,
                ):
                    if subject != "":
                        track_names.append(f"{cage}_{subject}")
                    else:
                        track_names.append(f"{cage}")
                break

            metadata_key_dict = {}
            for one_key in user_meta_data.keys():
                if one_key.endswith("mouse_ID_m1"):
                    metadata_key_dict["mouse_ID_m1"] = one_key
                elif one_key.endswith("mouse_ID_m2"):
                    metadata_key_dict["mouse_ID_m2"] = one_key
                elif one_key.endswith("cage_ID_m1"):
                    metadata_key_dict["cage_ID_m1"] = one_key
                elif one_key.endswith("cage_ID_m2"):
                    metadata_key_dict["cage_ID_m2"] = one_key

            if user_meta_data[metadata_key_dict["mouse_ID_m1"]] != "":
                track_names.append(
                    f"{user_meta_data[metadata_key_dict['cage_ID_m1']]}_{user_meta_data[metadata_key_dict['mouse_ID_m1']]}"
                )
            else:
                track_names.append(f"{user_meta_data[metadata_key_dict['cage_ID_m1']]}")

            if user_meta_data[metadata_key_dict["mouse_ID_m2"]] != "":
                if user_meta_data[metadata_key_dict["cage_ID_m2"]] != "":
                    track_names.append(
                        f"{user_meta_data[metadata_key_dict['cage_ID_m2']]}_{user_meta_data[metadata_key_dict['mouse_ID_m2']]}"
                    )
            elif user_meta_data[metadata_key_dict["cage_ID_m2"]] != "":
                track_names.append(f"{user_meta_data[metadata_key_dict['cage_ID_m2']]}")
            break

    return track_names


def extract_skeleton_nodes(
    skeleton_loc: str = None, skeleton_arena_bool: bool = False
) -> list:
    """
    Description
    ----------
    This function extracts names of skeleton nodes
    from the SLEAP .json file.

    NB: By default, the skeletons are read from the files
    located in the _config directory of the repo.
    ----------

    Parameters
    ----------
    skeleton_loc (str)
        The directory where skeletons can be found.
    skeleton_arena_bool (bool)
        If true, the function extracts the arena nodes; defaults to False.
    ----------

    Returns
    ----------
    skeleton_nodes (list)
        SLEAP skeleton node names.
    ----------
    """

    with open(skeleton_loc) as json_file:
        skeleton = json.load(json_file)

    unsorted_node_list = []
    for dict_idx, dict_id in enumerate(skeleton["links"]):
        if dict_idx == 0:
            unsorted_node_list.append(dict_id["source"]["py/state"]["py/tuple"][0])
            unsorted_node_list.append(dict_id["target"]["py/state"]["py/tuple"][0])
        elif "py/state" in dict_id["target"].keys():
            unsorted_node_list.append(dict_id["target"]["py/state"]["py/tuple"][0])

    sorting_key_list = []
    for node_dict in skeleton["nodes"]:
        raw_position_value = node_dict["id"]["py/id"]
        if raw_position_value == 1 or raw_position_value == 2:
            sorting_key_list.append(raw_position_value - 1)
        else:
            sorting_key_list.append(raw_position_value - 2)

    skeleton_nodes = [unsorted_node_list[n] for n in sorting_key_list]
    if skeleton_arena_bool:
        skeleton_nodes = [
            item if idx < 4 else f"ch_{item}" for idx, item in enumerate(skeleton_nodes)
        ]

    return skeleton_nodes


def redefine_cage_reference_nodes(
    arena_input_data: np.ndarray = None, node_list_indices: list = None
) -> np.ndarray:
    """
    Description
    ----------
    This function extracts names of skeleton nodes
    from the SLEAP .json file.

    ----------

    Parameters
    ----------
    arena_input_data (np.ndarray)
        3D arena data.
    node_list_indices (list)
        Indices of the arena nodes.
    ----------

    Returns
    ----------
    (np.ndarray)
        3D arena data with the arena corners.
    ----------
    """

    cage_corner_first = arena_input_data[0, 0, node_list_indices[0], :]
    cage_corner_second = arena_input_data[0, 0, node_list_indices[1], :]
    cage_corner_third = arena_input_data[0, 0, node_list_indices[2], :]
    cage_corner_fourth = arena_input_data[0, 0, node_list_indices[3], :]
    return np.vstack(
        tup=(
            cage_corner_first,
            cage_corner_second,
            cage_corner_third,
            cage_corner_fourth,
        )
    )


def rotate_x(data: np.ndarray = None, theta: float = None) -> np.ndarray:
    """
    Description
    ----------
    This function rotates data around the X axis.
    ----------

    Parameters
    ----------
    data (np.ndarray)
        3D arena/mouse data.
    theta (float)
        Angle of rotation in radians.
    ----------

    Returns
    ----------
    (np.ndarray)
        Rotated data.
    ----------
    """

    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta), math.sin(theta)],
            [0, -math.sin(theta), math.cos(theta)],
        ]
    )
    return np.matmul(data, rotation_matrix)


def rotate_y(data: np.ndarray = None, theta: float = None) -> np.ndarray:
    """
    Description
    ----------
    This function rotates data around the Y axis.
    ----------

    Parameters
    ----------
    data (np.ndarray)
        3D arena/mouse data.
    theta (float)
        Angle of rotation in radians.
    ----------

    Returns
    ----------
    (np.ndarray)
        Rotated data.
    ----------
    """

    rotation_matrix = np.array(
        [
            [math.cos(theta), 0, -math.sin(theta)],
            [0, 1, 0],
            [math.sin(theta), 0, math.cos(theta)],
        ]
    )
    return np.matmul(data, rotation_matrix)


def rotate_z(data: np.ndarray = None, theta: float = None) -> np.ndarray:
    """
    Description
    ----------
    This function rotates data around the Z axis.
    ----------

    Parameters
    ----------
    data (np.ndarray)
        3D arena/mouse data.
    theta (float)
        Angle of rotation in radians.
    ----------

    Returns
    ----------
    (np.ndarray)
        Rotated data.
    ----------
    """

    rotation_matrix = np.array(
        [
            [math.cos(theta), math.sin(theta), 0],
            [-math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return np.matmul(data, rotation_matrix)


class ConvertTo3D:
    def __init__(
        self,
        root_directory: str = None,
        input_parameter_dict: dict = None,
        message_output: callable = None,
    ) -> None:
        """
        Initializes the ConvertTo3D class.

        Parameter
        ---------
        root_directory (str)
            Root directory for data; defaults to None.
        input_parameter_dict (dict)
            Processing parameters; defaults to None.
        message_output (function)
            Output messages; defaults to None.

        Returns
        -------
        -------
        """

        if input_parameter_dict is None:
            with open(
                pathlib.Path(__file__).parent
                / "_parameter_settings/processing_settings.json"
            ) as json_file:
                self.input_parameter_dict = json.load(json_file)["anipose_operations"][
                    "ConvertTo3D"
                ]
        else:
            self.input_parameter_dict = input_parameter_dict["anipose_operations"][
                "ConvertTo3D"
            ]

        if root_directory is None:
            with open(
                pathlib.Path(__file__).parent
                / "_parameter_settings/processing_settings.json"
            ) as json_file:
                self.root_directory = json.load(json_file)["anipose_operations"][
                    "root_directory"
                ]
        else:
            self.root_directory = root_directory

        if message_output is None:
            self.message_output = print
        else:
            self.message_output = message_output

        self.session_root_joint_date_dir = ""
        self.session_root_name = ""
        for one_object in os.listdir(f"{self.root_directory}{os.sep}video"):
            if (
                os.path.isdir(os.path.join(self.root_directory, "video", one_object))
                and "_" not in one_object
            ):
                self.session_root_joint_date_dir = os.path.join(
                    self.root_directory, "video", one_object
                )
                self.session_root_name = one_object

        self.app_context_bool = is_gui_context()

    def sleap_file_conversion(self) -> None:
        """
        Description
        ----------
        This function runs the SLP to H5 conversion in parallel
        for all videos recorded.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        .h5 analysis files
        ----------
        """

        self.message_output(
            f"SLEAP file conversion started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}"
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        if os.name == "nt":
            command_addition = "cmd /c "
            shell_usage_bool = False
        else:
            command_addition = ''
            shell_usage_bool = True

        conversion_subprocesses = []
        for cam_directory in os.listdir(self.session_root_joint_date_dir):
            if os.path.isdir(
                os.path.join(self.session_root_joint_date_dir, cam_directory)
            ):
                for one_file in os.listdir(
                    os.path.join(self.session_root_joint_date_dir, cam_directory)
                ):
                    if one_file.endswith(".slp"):
                        conversion_subp = subprocess.Popen(
                            args=f'''{command_addition}uvx --from sleap[nn] sleap-convert --format analysis -o "{one_file[:-3]}analysis.h5" "{one_file}"''',
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT,
                            cwd=os.path.join(
                                self.session_root_joint_date_dir, cam_directory
                            ),
                            shell=shell_usage_bool,
                        )
                        conversion_subprocesses.append(conversion_subp)

        while True:
            status_poll = [query_subp.poll() for query_subp in conversion_subprocesses]
            if any(elem is None for elem in status_poll):
                smart_wait(app_context_bool=self.app_context_bool, seconds=1)
            else:
                break

    def conduct_anipose_calibration(self) -> None:
        """
        Description
        ----------
        This method conducts the calibration routine for SLEAP Anipose.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        Calibration files:
            cboard.toml
            calibration.toml
            calibration.metadata.h5
            reprojection_histogram.png
        ----------
        """

        self.message_output(
            f"ANIPOSE calibration started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}"
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        if not self.input_parameter_dict["conduct_anipose_calibration"][
            "board_provided_bool"
        ]:
            sleap_anipose.draw_board(
                board_name=f"{self.session_root_joint_date_dir}{os.sep}{self.session_root_name}_charuco_8x11.png",
                board_x=self.input_parameter_dict["conduct_anipose_calibration"][
                    "board_xy"
                ][0],
                board_y=self.input_parameter_dict["conduct_anipose_calibration"][
                    "board_xy"
                ][1],
                square_length=self.input_parameter_dict["conduct_anipose_calibration"][
                    "square_len"
                ],
                marker_length=self.input_parameter_dict["conduct_anipose_calibration"][
                    "marker_len_bits"
                ][0],
                marker_bits=self.input_parameter_dict["conduct_anipose_calibration"][
                    "marker_len_bits"
                ][1],
                dict_size=self.input_parameter_dict["conduct_anipose_calibration"][
                    "dict_size"
                ],
                img_width=self.input_parameter_dict["conduct_anipose_calibration"][
                    "img_width_height"
                ][0],
                img_height=self.input_parameter_dict["conduct_anipose_calibration"][
                    "img_width_height"
                ][1],
                save=f"{self.session_root_joint_date_dir}{os.sep}{self.session_root_name}_cboard.toml",
            )

        sleap_anipose.calibrate(
            session=self.session_root_joint_date_dir,
            board=f"{self.session_root_joint_date_dir}{os.sep}{self.session_root_name}_cboard.toml",
            calib_fname=f"{self.session_root_joint_date_dir}{os.sep}{self.session_root_name}_calibration.toml",
            metadata_fname=f"{self.session_root_joint_date_dir}{os.sep}{self.session_root_name}_calibration.metadata.h5",
            histogram_path=f"{self.session_root_joint_date_dir}{os.sep}{self.session_root_name}_reprojection_histogram.png",
            reproj_path=self.session_root_joint_date_dir,
        )

    def conduct_anipose_triangulation(self) -> None:
        """
        Description
        ----------
        This method runs the 3D triangulation routine for SLEAP Anipose.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        points3d (h5 file)
            3D triangulated point h5 file,
            shape: (N_FRAMES, N_ANIMALS, N_NODES, N_DIMENSIONS).
        ----------
        """

        self.message_output(
            f"ANIPOSE triangulation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}"
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        calibration_dir_search = glob.glob(
            pathname=os.path.join(
                f"{self.input_parameter_dict['conduct_anipose_triangulation']['calibration_file_loc']}{os.sep}**",
                "*_calibration.toml*",
            ),
            recursive=True,
        )
        if len(calibration_dir_search) == 0:
            self.message_output(
                "Calibration directory not found. Please run calibration first and provide a correct path."
            )
        else:
            calibration_toml_file = calibration_dir_search[0]

            if not self.input_parameter_dict["conduct_anipose_triangulation"][
                "triangulate_arena_points_bool"
            ]:
                none_hyperparam_bool = False
                if (
                    self.input_parameter_dict["conduct_anipose_triangulation"][
                        "frame_restriction"
                    ]
                    is None
                ):
                    with open(
                        glob.glob(
                            pathname=os.path.join(
                                f"{self.root_directory}{os.sep}video",
                                "*_camera_frame_count_dict.json*",
                            )
                        )[0]
                    ) as frame_count_infile:
                        camera_frame_count_dict = json.load(frame_count_infile)
                        self.input_parameter_dict["conduct_anipose_triangulation"][
                            "frame_restriction"
                        ] = [
                            0,
                            int(camera_frame_count_dict["total_frame_number_least"]),
                        ]
                        none_hyperparam_bool = True

                sleap_anipose.triangulate(
                    p2d=self.session_root_joint_date_dir,
                    calib=calibration_toml_file,
                    frames=tuple(
                        self.input_parameter_dict["conduct_anipose_triangulation"][
                            "frame_restriction"
                        ]
                    ),
                    excluded_views=tuple(
                        self.input_parameter_dict["conduct_anipose_triangulation"][
                            "excluded_views"
                        ]
                    ),
                    ransac=self.input_parameter_dict["conduct_anipose_triangulation"][
                        "ransac_bool"
                    ],
                    fname=f"{self.session_root_joint_date_dir}{os.sep}{self.session_root_name}_points3d.h5",
                    disp_progress=self.input_parameter_dict[
                        "conduct_anipose_triangulation"
                    ]["display_progress_bool"],
                    constraints=self.input_parameter_dict[
                        "conduct_anipose_triangulation"
                    ]["rigid_body_constraints"],
                    constraints_weak=self.input_parameter_dict[
                        "conduct_anipose_triangulation"
                    ]["weak_body_constraints"],
                    scale_smooth=self.input_parameter_dict[
                        "conduct_anipose_triangulation"
                    ]["smooth_scale"],
                    scale_length=self.input_parameter_dict[
                        "conduct_anipose_triangulation"
                    ]["weight_rigid"],
                    scale_length_weak=self.input_parameter_dict[
                        "conduct_anipose_triangulation"
                    ]["weight_weak"],
                    reproj_error_threshold=self.input_parameter_dict[
                        "conduct_anipose_triangulation"
                    ]["reprojection_error_threshold"],
                    reproj_loss=self.input_parameter_dict[
                        "conduct_anipose_triangulation"
                    ]["regularization_function"],
                    n_deriv_smooth=self.input_parameter_dict[
                        "conduct_anipose_triangulation"
                    ]["n_deriv_smooth"],
                )

                if none_hyperparam_bool:
                    self.input_parameter_dict["conduct_anipose_triangulation"][
                        "frame_restriction"
                    ] = None

            else:
                none_hyperparam_bool = False
                if (
                    self.input_parameter_dict["conduct_anipose_triangulation"][
                        "frame_restriction"
                    ]
                    is None
                ):
                    self.input_parameter_dict["conduct_anipose_triangulation"][
                        "frame_restriction"
                    ] = [0, 1]
                    none_hyperparam_bool = True

                sleap_anipose.triangulate(
                    p2d=self.session_root_joint_date_dir,
                    calib=calibration_toml_file,
                    frames=tuple(
                        self.input_parameter_dict["conduct_anipose_triangulation"][
                            "frame_restriction"
                        ]
                    ),
                    excluded_views=tuple(
                        self.input_parameter_dict["conduct_anipose_triangulation"][
                            "excluded_views"
                        ]
                    ),
                    fname=f"{self.session_root_joint_date_dir}{os.sep}{self.session_root_name}_points3d.h5",
                    disp_progress=self.input_parameter_dict[
                        "conduct_anipose_triangulation"
                    ]["display_progress_bool"],
                    reproj_error_threshold=self.input_parameter_dict[
                        "conduct_anipose_triangulation"
                    ]["reprojection_error_threshold"],
                )

                if none_hyperparam_bool:
                    self.input_parameter_dict["conduct_anipose_triangulation"][
                        "frame_restriction"
                    ] = None

    def translate_rotate_metric(self, **kwargs) -> None:
        """
        Description
        ----------
        This method translates and rotates the 3D points file, and converts units to meters.
        ----------

        Parameters
        ----------
        ----------

        Returns
        ----------
        translated_rotated_metric (h5 file)
            3D translated rotated and metric point h5 file,
            shape: (N_FRAMES, N_ANIMALS, N_NODES, N_DIMENSIONS).
        ----------
        """

        self.message_output(
            f"Translation, rotation and metric conversion started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}.{datetime.now().second:02d}"
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        session_idx = (
            kwargs["session_idx"]
            if "session_idx" in kwargs and isinstance(kwargs["session_idx"], int)
            else 0
        )

        # get recording frame rate
        with open(
            glob.glob(
                pathname=os.path.join(
                    f"{self.root_directory}{os.sep}video",
                    "*_camera_frame_count_dict.json*",
                )
            )[0]
        ) as frame_count_infile:
            recording_frame_rate = json.load(frame_count_infile)[
                "median_empirical_camera_sr"
            ]

        # load original arena data
        arena_data_original_h5 = glob.glob(
            pathname=f"{self.input_parameter_dict['translate_rotate_metric']['original_arena_file_loc']}{os.sep}**{os.sep}*_points3d.h5*",
            recursive=True,
        )[0]
        arena_data_original_h5_dir = os.path.dirname(arena_data_original_h5)
        arena_data_original_h5_file = os.path.basename(arena_data_original_h5)
        with h5py.File(arena_data_original_h5, mode="r") as h5_file_arena:
            arena_data = np.array(h5_file_arena["tracks"], dtype="float64")

        arena_nodes = extract_skeleton_nodes(
            skeleton_loc=pathlib.Path(__file__).parent
            / "_config/playpen_skeleton.json",
            skeleton_arena_bool=True,
        )

        # convert unit of measurement to meters
        node_list_indices = [
            arena_nodes.index("North"),
            arena_nodes.index("East"),
            arena_nodes.index("South"),
            arena_nodes.index("West"),
        ]
        arena_corners = redefine_cage_reference_nodes(
            arena_input_data=arena_data, node_list_indices=node_list_indices
        )
        mean_playpen_edge = np.nanmean(
            [
                np.linalg.norm(arena_corners[0, :] - arena_corners[1, :]),
                np.linalg.norm(arena_corners[0, :] - arena_corners[3, :]),
                np.linalg.norm(arena_corners[2, :] - arena_corners[1, :]),
                np.linalg.norm(arena_corners[2, :] - arena_corners[3, :]),
            ]
        )
        metric_conversion_coefficient = (
            self.input_parameter_dict["translate_rotate_metric"]["static_reference_len"]
            / mean_playpen_edge
        )
        arena_data = arena_data * metric_conversion_coefficient

        # translate data relative to playpen midpoint
        arena_corners = redefine_cage_reference_nodes(
            arena_input_data=arena_data, node_list_indices=node_list_indices
        )
        cage_midpoint = np.nanmean(arena_corners, axis=0)
        arena_data = arena_data - cage_midpoint
        arena_corners = redefine_cage_reference_nodes(
            arena_input_data=arena_data, node_list_indices=node_list_indices
        )

        # rotate arena data
        z_theta = -math.atan2(
            arena_corners[0, 1] - arena_corners[2, 1],
            arena_corners[0, 0] - arena_corners[2, 0],
        )
        arena_data_temp = arena_data.copy()
        arena_data = rotate_z(arena_data_temp, z_theta)
        arena_corners = redefine_cage_reference_nodes(
            arena_input_data=arena_data, node_list_indices=node_list_indices
        )

        arena_data_temp = arena_data.copy()
        y_theta = (
            math.atan2(
                arena_corners[0, 2] - arena_corners[2, 2],
                arena_corners[0, 0] - arena_corners[2, 0],
            )
            + math.pi
        )
        arena_data = rotate_y(arena_data_temp, y_theta)
        arena_corners = redefine_cage_reference_nodes(
            arena_input_data=arena_data, node_list_indices=node_list_indices
        )

        arena_data_temp = arena_data.copy()
        x_theta = -math.atan2(
            arena_corners[1, 2] - arena_corners[3, 2],
            arena_corners[1, 1] - arena_corners[3, 1],
        )
        arena_data = rotate_x(arena_data_temp, x_theta)

        arena_data_temp = arena_data.copy()
        z_theta_extra = -math.pi / 4
        arena_data = rotate_z(arena_data_temp, z_theta_extra)

        # take same distance to all corners and correct corners to be the outer edge of rail instead of inside corner
        arena_center_out_distance = np.max(
            [
                np.abs(
                    np.nanmin(
                        arena_data[
                            0,
                            0,
                            [
                                arena_nodes.index("North"),
                                arena_nodes.index("West"),
                                arena_nodes.index("South"),
                                arena_nodes.index("East"),
                            ],
                            :2,
                        ]
                    )
                ),
                np.nanmax(
                    arena_data[
                        0,
                        0,
                        [
                            arena_nodes.index("North"),
                            arena_nodes.index("West"),
                            arena_nodes.index("South"),
                            arena_nodes.index("East"),
                        ],
                        :2,
                    ]
                ),
            ]
        )

        arena_data[0, 0, arena_nodes.index("East"), :] = [
            (
                np.sign(arena_data[0, 0, arena_nodes.index("East"), 0])
                * arena_center_out_distance
            )
            + 0.025,
            (
                np.sign(arena_data[0, 0, arena_nodes.index("East"), 1])
                * arena_center_out_distance
            )
            + 0.025,
            0,
        ]

        arena_data[0, 0, arena_nodes.index("West"), :] = [
            (
                np.sign(arena_data[0, 0, arena_nodes.index("West"), 0])
                * arena_center_out_distance
            )
            - 0.025,
            (
                np.sign(arena_data[0, 0, arena_nodes.index("West"), 1])
                * arena_center_out_distance
            )
            - 0.025,
            0,
        ]

        arena_data[0, 0, arena_nodes.index("North"), :] = [
            (
                np.sign(arena_data[0, 0, arena_nodes.index("North"), 0])
                * arena_center_out_distance
            )
            - 0.025,
            (
                np.sign(arena_data[0, 0, arena_nodes.index("North"), 1])
                * arena_center_out_distance
            )
            + 0.025,
            0,
        ]

        arena_data[0, 0, arena_nodes.index("South"), :] = [
            (
                np.sign(arena_data[0, 0, arena_nodes.index("South"), 0])
                * arena_center_out_distance
            )
            + 0.025,
            (
                np.sign(arena_data[0, 0, arena_nodes.index("South"), 1])
                * arena_center_out_distance
            )
            - 0.025,
            0,
        ]

        if (
            self.input_parameter_dict["translate_rotate_metric"][
                "save_transformed_data"
            ]
            == "arena"
        ):
            with h5py.File(
                name=f"{arena_data_original_h5_dir}{os.sep}{arena_data_original_h5_file[:-3]}_translated_rotated_metric.h5",
                mode="w",
            ) as h5_file_write:
                h5_file_write.create_dataset(name="tracks", data=arena_data)
                h5_file_write.create_dataset(name="node_names", data=arena_nodes)

            self.message_output(
                f"Triangulated arena file has shape: {arena_data.shape}"
            )
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        elif (
            self.input_parameter_dict["translate_rotate_metric"][
                "save_transformed_data"
            ]
            == "animal"
        ):
            with h5py.File(
                name=f"{self.session_root_joint_date_dir}{os.sep}{self.session_root_name}_points3d.h5",
                mode="r",
            ) as h5_file_mouse:
                mouse_data = np.array(h5_file_mouse["tracks"], dtype="float64")

            mouse_nodes = extract_skeleton_nodes(
                skeleton_loc=pathlib.Path(__file__).parent
                / "_config/mouse_skeleton.json",
                skeleton_arena_bool=False,
            )

            mouse_data = mouse_data * metric_conversion_coefficient
            mouse_data = mouse_data - cage_midpoint

            mouse_data_temp = mouse_data.copy()
            mouse_data = rotate_z(mouse_data_temp, z_theta)

            mouse_data_temp = mouse_data.copy()
            mouse_data = rotate_y(mouse_data_temp, y_theta)

            mouse_data_temp = mouse_data.copy()
            mouse_data = rotate_x(mouse_data_temp, x_theta)

            mouse_data_temp = mouse_data.copy()
            mouse_data = rotate_z(mouse_data_temp, z_theta_extra)

            # set all negative mouse z-coordinates to 0
            negative_z_indices = np.where(mouse_data[:, :, :, 2] < 0)
            if negative_z_indices[0].size > 0:
                mouse_data[
                    negative_z_indices[0],
                    negative_z_indices[1],
                    negative_z_indices[2],
                    2,
                ] = 0

            mouse_track_names = find_mouse_names(root_directory=self.root_directory)

            with h5py.File(
                name=f"{self.session_root_joint_date_dir}{os.sep}{self.session_root_name}_points3d_translated_rotated_metric.h5",
                mode="w",
            ) as h5_file_write:
                h5_file_write.create_dataset(name="tracks", data=mouse_data)
                h5_file_write.create_dataset(name="node_names", data=mouse_nodes)
                h5_file_write.create_dataset(name="track_names", data=mouse_track_names)
                h5_file_write.create_dataset(
                    name="experimental_code",
                    data=self.input_parameter_dict["translate_rotate_metric"][
                        "experimental_codes"
                    ][session_idx],
                )
                h5_file_write.create_dataset(
                    name="recording_frame_rate", data=recording_frame_rate
                )

            self.message_output(
                f"Triangulated mouse file has shape: {mouse_data.shape}"
            )
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)

            if self.input_parameter_dict["translate_rotate_metric"][
                "delete_original_h5"
            ]:
                os.remove(
                    f"{self.session_root_joint_date_dir}{os.sep}{self.session_root_name}_points3d.h5"
                )
