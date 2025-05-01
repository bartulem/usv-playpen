.. _Record:

Record
======
This page explains how to use the data recording functionalities in the usv-playpen GUI.

Calibrate (introduction)
------------------------
In *usv-playpen*, camera calibration has a two-fold purpose: (1) learning the spatial
relationship between the used cameras, (2) capturing positions of microphones and
arena corners.

In order to reconstruct mouse body key-points in 3D, one first needs to ensure that the
cameras understand how they are positioned relative to one another. This is done by displaying
a ChArUco board in the field of view of all cameras, ideally in many orientations, and then calculating the reprojection
error of specific markers.

A brief introduction on the uses of ChArUco boards can be found `here <https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html>`_.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/charuco.png
   :width: 650
   :height: 400
   :align: center
   :alt: ChArUco board

.. raw:: html

   <br>

Calibrate (preparation)
-----------------------

Create a ChArUco board
^^^^^^^^^^^^^^^^^^^^^^

OpenCV provides a function to create a ChArUco board. This is implemented through SLEAP Anipose and available in the GUI.
A thing to consider is the size of the board. The larger the board, the more markers it contains, and the more accurate the
calibration will be. The user is responsible for printing the board and attaching it to a flat surface. By default, the board
has the following characteristics (found in */src/_parameter_settings/process_settings.json*):

.. code-block:: json

    "calibrate_anipose": {
        "board_provided_bool": false,
        "board_xy": [
          8,
          11
        ],
        "square_len": 24,
        "marker_len_bits": [
          18.75,
          4
        ],
        "dict_size": 1000,
        "img_width_height": [
          2100,
          2970
        ]
      },

Clear arena and position IR-markers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/placing_markers.gif
   :width: 225
   :height: 400
   :align: left
   :alt: Corner marker

Arena corners are fitted with four laser-cut acrylic parts that are proped vertically to discourage mice from climbing up. Each part is a three-piece design consisting of the main vertical piece, a top cover and
screw that secures the top to the t-brackets. Secured parts should not wobble when you apply perpendicular pressure. It should press tightly against the floor plate. You can remove these parts by
loosening the screws and removing the pieces one by one. Try not to push any of these acrylic parts against the floor while removing, since they can break.

After removing the corner parts, each of the four mesh
barriers can be slid out from the metal frame by gently pulling up (you can bend them slightly to avoid obstacles like SYNC LEDs). The important consideration in this process is to **remain mindful of camera positions**,
such that you avoid bumping into them, as changing camera positions will require a new calibration procedure.

The objective of this procedure is to improve the range of the ChArUco board and to enable microphone visibility. Tracking microphone positions is
necessary for localizing sound sources, and tracking arena corners helps us establish the boundaries of the environment. To simplify tracking of the
corners, which may not be visible to the cameras, we use IR-retroreflective markers. One can place four half-spherical retroreflectors to four
corners of the arena (see example video above): they now establish the approximate boundary of the arena.

To expedite execution, place the physical ChArUco board inside the arena (see example video above), and check that all IR reflectors are turned on: you want to make sure
they are pointing roughly into the center of the arena. Finally, check camera availability and visibility in the Motif web interface (see image below for comparison).

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/ir_reflectors_difference.png
   :width: 800
   :height: 320
   :align: center
   :alt: IR Reflectors Difference

.. raw:: html

   <br>

Calibrate (execution)
---------------------
In the GUI main window, select experimenter name from the dropdown menu and click *Record*.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/calibration_step_1.png
   :align: center
   :alt: Calibration Step 1

.. raw:: html

   <br>

Depending on the choice of experimenter name, you can see file server directory destinations of files created during recording. You can naturally change this setting as you please. Several important details, however, are present in the section below. For a camera Calibration session, we choose not to conduct an audio recording, but to conduct video calibration. By default, calibration duration is 5 minutes long and the recording of the empty arena after it is 1 minute long, but these are arbitrary and should be adjusted to particular needs. For calibration is also not necessary to disable the ethernet connection.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/calibration_step_2.png
   :align: center
   :alt: Calibration Step 2

.. raw:: html

   <br>

One can skip the Audio Settings step as it is not relevant here, and adjust Video Settings. Each video recording is associated with a particular metadata file and below you can see an example of how one might fill out the metadata form for calibration. On the left, you can use a slider to choose the acquisition frame rate of cameras during calibration. By default, this is set to 10 fps, and generally lower values are better for calibration.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/calibration_step_3.png
   :align: center
   :alt: Calibration Step 3

.. raw:: html

   <br>

Finally, when ready to head over to the arena and move the calibration board, simply click the *Calibrate* button. You have several seconds before the video starts recording.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/calibration_step_4.png
   :align: center
   :alt: Calibration Step 4

.. raw:: html

   <br>

The video below is a sped-up version of an actual calibration and can be consulted for reference.

.. image:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/calibration_process.gif
   :width: 225
   :height: 400
   :align: left
   :alt: Calibration Example

It is good practice to be thorough and move the ChArUco board around the arena, so that all cameras can see it. The more markers are visible to the cameras, the better the calibration will be. You can also rotate the board in your hands to ensure that all markers are visible from different angles. The calibration process is not very sensitive to the distance of the board from the cameras, but it is important to keep it in focus.

Make sure you cover all sides and corners, but also move the board slightly in the vertical dimension, without moving too far from the floor. Change the angle of the board relative to the cameras freely, but keep in mind that extreme orientations may not be captured by the cameras at all. Moving the board over the microphones is not problematic, as long as it is not out of the range of the cameras or displacing the microphones.

When Calibration is complete, you can leave the board on the floor and click the *Record* button, which will capture a minute long video of the empty arena. You do not want to move around in the arena space during this recording. Upon completion, the data will be copied over to the directories/fileserver(s) you selected previously, *e.g.*, F:/Bartul/Data/20250430_141750 and there will be two subdirectories: *sync* and *video*. In the *video* subdirectory, you will find Nx (N = number of cameras) calibration subdirectories (containing 5 minute calibration videos) and Nx recording subdirectories (containing the 1 minute video post calibration).


Calibrate (assessment)
----------------------
To assess the quality of the calibration, you first click the *Process* button on the GUI main display.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/calibration_step_5.png
   :align: center
   :alt: Calibration Step 5

.. raw:: html

   <br>

In the *Root directories for processing* window, either write or c/p the path to the directory of the calibration session you just recorded. Select *Run video re-encoding* adn change the *Concatenation name* to 000000. Finally, select *Run AP Calibration*. Hit *Next*, and *Process*. In the terminal/powershell, you should be able to see the amount of CharUco Boards detected by reprojection on each camera, as progress bars will appear.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/calibration_step_6.png
   :align: center
   :alt: Calibration Step 6

.. raw:: html

   <br>

When Calibration is done, if you navigate to, *e.g.*, F:/Bartul/Data/20250430_141750/20250430141750/video, you will find, among others, a *20250430141750_calibration.toml* file and a *20250430141750_reprojection_histogram.png* file. The histogram should display the reprojection error diminishing steeply (see image below for example), highly suggestive of an effective calibration.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/reprojection_histogram_example.png
   :align: center
   :width: 500
   :height: 375
   :alt: Reprojection Example

.. raw:: html

   <br>

Record (general settings)
-------------------------
Firstly, you want to remove the retro-reflective markers, install the screen doors, and secure four corners with custom covers. Check that IR-reflectors are all connected, and the overhead light is turned to warm light and the intensity is low. If necessary, also clean the surface of the floor the animals walk on.

In the GUI main window, select experimenter name from the dropdown menu and click *Record*.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/calibration_step_1.png
   :align: center
   :alt: Recording Step 0

.. raw:: html

   <br>

On the next page, you can set some basic parameters:

* **Avisoft Recorder directory** : this is the directory which contains the *rec_usgh.exe* file
* **Avisoft base directory** : this is the local directory where the recordings will be saved
* **Coolterm directory** : this is the local directory where the Arduino serial terminal outputs will be saved
* **File destination(s) Linux** : this is the directory on both video PCs where the file server is mounted
* **File destination(s) Windows** : this is the directory on the audio PC where the file server is mounted
* **Conduct AUDIO recording** :  if *Yes*, the audio recording will be conducted; if *No*, only video will be recorded
* **Conduct VIDEO calibration** : if *Yes*, the video calibration will be conducted
* **Disable ethernet connection** : if *Yes*, the ethernet connection will be disabled during the recording
* **Video session duration (min)** : total duration of the recording session
* **Calibration duration (min)** : duration of the calibration session
* **Ethernet network ID** : this is the ID of the ethernet network
* **Notify e-mail(s) of PC usage** : this is the e-mail address of the person who will be notified of start and end of PC usage

In the example below, one would be doing a 20 minute audio and video recording without calibration. When ready, click *Next*.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/recording_step_1.png
   :align: center
   :alt: Recording Step 1

.. raw:: html

   <br>


Record (audio settings)
-----------------------
In the *Audio Settings* window, you can set the parameters for the audio recording. Avisoft Recorder USGH has a relatively complex set of options and using the default ones is probably best because they provide the best stability, although bugs can still occur. The *Audio settings* tab contains many parameters that hardly ever need changing. Of these, you might want to pay attention to three:

* **cpu_priority**: Windows option that regulates resource management based on the importance of the process
* **cpu_affinity**: Windows option that regulates the CPU core on which the process will run
* **usghflags**: audio devices operate in SYNC mode (1574) or separately (1862)

In the example below, one would be setting the Audio Recorder USGH to run on processor *6* with *high* priority, and the devices are operating in sync mode. When ready, click *Next*.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/recording_step_2.png
   :align: center
   :alt: Recording Step 2

.. raw:: html

   <br>

Record (video settings)
-----------------------

In the *Video Settings* window, you can set the parameters for the video recording:

* **Browser** : this is the browser you want to use for viewing video recordings live
* **Camera(s) to use** : serial numbers of cameras you want to use in the recording
* **Recording codec** : this is the codec (video quality) you want to use for the recording
* **Monitor recording** : if *Yes*, monitor recording on this PC **(incompatible with disabling ethernet!)**
* **Monitor ONE camera** : If *Yes*, monitor only one camera in browser
* **ONE camera serial** : if monitoring one camera, this is the serial number of the camera you want to monitor
* **Delete post copy** : if *Yes*, the video files will be deleted from the video PCs after copying to the file server
* **Calibration fps** : calibration frame rate (fps) of the cameras
* **Recording fps** : recording frame rate (fps) of the cameras
* **Particular camera settings** : exposure time and gain setting for every available camera
* **Metadata** : metadata for the recording session

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/recording_step_3.png
   :align: center
   :alt: Recording Step 3

.. raw:: html

   <br>

Clicking *Next* saves all your settings to the */_config/behavioral_experiments_settings.toml* file. You should also observe how upon starting another recording, **all settings which you set previously will be automatically loaded**. When the mice are in the arena and the doors are closed, click *Record*.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/recording_step_4.png
   :align: center
   :alt: Recording Step 4

.. raw:: html

   <br>

Record (common issues)
----------------------
Placeholder text.