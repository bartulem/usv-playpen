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

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/retroreflective_marker.gif
   :width: 400
   :height: 500
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

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/ir_reflectors_diff.png
   :width: 800
   :height: 320
   :align: center
   :alt: IR Reflectors Difference

Calibrate (execution)
---------------------
In the GUI main window, select experimenter name and click *Record*.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/cal1.png
   :align: center
   :alt: Calibration Step 1

| Depending on the choice of experimenter name, you can see directory destinations of files created during recording. For a camera Calibration session, we choose not to conduct an audio recording, but to conduct video calibration. By default, calibration duration is 5 minutes long and the recording of the empty arena after it is 1 minute long, but these are arbitrary and should be adjusted to particular needs. For calibration is also not necessary to disable the ethernet connection.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/cal2.png
   :align: center
   :alt: Calibration Step 2

| One can skip the Audio Settings step as it is not relevant here, and adjust Video Settings. Each video recording is associated with a particular metadata file and below you can see an example of how one might fill out the metadata form for calibration. On the left, you can use a slider to choose the acquisition frame rate of cameras during calibration. By default, this is set to 10 fps, and generally lower values are better for calibration.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/cal3.png
   :align: center
   :alt: Calibration Step 3

| Finally, when ready to head over to the arena and move the calibration board, simply click the *Calibrate* button. You have several seconds before the video starts recording.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/cal4.png
   :align: center
   :alt: Calibration Step 4

| The video below is a sped-up version of an actual calibration and can be consulted for reference.

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/calibration_.gif
   :width: 280
   :height: 500
   :align: left
   :alt: Calibration Example

It is good practice to be thorough and move the ChArUco board around the arena, so that all cameras can see it. The more markers are visible to the cameras, the better the calibration will be. You can also rotate the board in your hands to ensure that all markers are visible from different angles. The calibration process is not very sensitive to the distance of the board from the cameras, but it is important to keep it in focus.

| Make sure you cover all sides and corners, but also move the board slightly in the vertical dimension, without moving too far from the floor. Change the angle of the board relative to the cameras freely, but keep in mind that extreme orientations may not be captured by the cameras at all. Moving the board over the microphones is not problematic, as long as it is not out of the range of the cameras or displacing the microphones.

| When Calibration is complete, you can leave the board on the floor and click the *Record* button, which will capture a minute long video on the empty arena. You do not want to move around in the arena space during this recording.


Calibrate (assessment)
----------------------
Placeholder text.

Record (general settings)
-------------------------
Placeholder text.

Record (audio settings)
-----------------------
Placeholder text.

Record (video settings)
-----------------------
Placeholder text.

Record (common issues)
----------------------
Placeholder text.