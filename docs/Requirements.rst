.. _Requirements:

Requirements
============

This page explains how to set up your equipment and PCs before using the *usv-playpen* GUI for behavioral recordings.

Hardware Requirements
---------------------
Audio recording essentials
^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Dell Precision 3680 Tower Intel(R) Core(TM) i9-1900 CPU @ 2.00 GHz and 64 GB RAM <https://www.dell.com/en-us/shop/desktop-computers/precision-3680-tower-workstation/spd/precision-t3680-workstation>`_ (1x)
* `Avisoft UltraSoundGate Player 1216H (comes with SYNC cable) <hhttps://avisoft.com/ultrasoundgate/1216h/>`_ (2x)
* `Avisoft 40011 CM16/CMPA microphones (come with XLR-5 extension cables) <https://avisoft.com/ultrasound-microphones/cm16-cmpa/>`_ (24x)
* `Sound permeable mesh <https://www.mcmaster.com/catalog/131/470/9318T25>`_ (10ft)
* `Adhesive board for holding the mesh in place <https://www.amazon.com/BENECREAT-Self-Adhesive-Insulation-Containers-Protection/dp/B08DY8QD4Y?th=1>`_ (10x)

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/audio_recording_hardware.png
   :width: 800
   :height: 320
   :align: center
   :alt: Audio Hardware

.. raw:: html

   <br>

Video recording essentials
^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Dell Precision 3650 Tower <https://www.dell.com/en-us/shop/desktops-all-in-ones/precision-3650-tower-workstation/spd/precision-3650-workstation>`_ (2x)
* `PNY NVIDIA Quadro P2200 <https://www.pny.com/nvidia-quadro-p2200>`_ (2x)
* `USB 3.0 to PCIe x4 Gen 2 HBA U3X4-PCiE4XE111 <https://www.ioiusb.com/Host-Adapter/U3X4-PCIE4XE111.htm>`_ (2x)
* `15 Pin SATA Power Extension Cable <https://www.amazon.com/Extension-Extender-Adapter-Optical-Burners/dp/B07SXDXPFL/ref=sr_1_7?crid=ZIBDE5UR65VQ&keywords=sata+15&qid=1641544288&sprefix=sata+15%2Caps%2C167&sr=8-7>`_ (2x)
* `Cat5e Network Ethernet Cable (1G bps, 350 MHz) <https://www.amazon.com/Cmple-CAT5E-ETHERNET-Network-Cable/dp/B00B1TU3WY/ref=sr_1_1?crid=5PHLA498GSC4&dib=eyJ2IjoiMSJ9.cApl-5oXAZ73r65_nI_e4g.ObJUGm0zNRkTYfMs4VxxP8R_ap1_8v58SFKJZ2EBzdI&dib_tag=se&keywords=B00B1TU3WY&qid=1723062191&sprefix=b00b1tu3wy%2Caps%2C60&sr=8-1&th=1>`_ (2x)
* `BFS-U3-13Y3M-C <https://www.teledynevisionsolutions.com/products/blackfly-s-usb3/>`_ (5x)
* `USB3 Micro-B Locking Cable <https://www.teledynevisionsolutions.com/products/usb-3.1-locking-cable/?model=ACC-01-2300&segment=iis&vertical=machine%20vision/>`_ (5x)
* `Blackfly S/Blackfly 6-pin GPIO Cable <https://www.edmundoptics.com/p/blackflyreg-6-pin-gpio-hirose-connector-45m-cable/30350/>`_ (5x)
* `Camera mounting gimble <https://www.digikey.com/en/products/detail/panavise/851-00/2602033>`_ (5x)
* `Overhead lens 8mm or 8.5mm lens for 1/2" sensors HF8XA-5M <https://www.rmaelectronics.com/fujinon-hf8xa-5m/>`_ (1x)
* `Side lens 6mm lens for 1/2" sensors HF6XA-5M <https://www.rmaelectronics.com/fujinon-hf6xa-5m/>`_ (4x)
* `IR filter LP780-37.5: FILTER NIR LONGPASS M37.5 <https://midopt.com/filters/lp780/>`_ (5x)
* `Light phidget <https://phidgets.com/?tier=3&catid=8&pcid=6&prodid=707>`_ (1x)
* `Humidity/temperature phidget <https://phidgets.com/?tier=3&catid=14&pcid=12&prodid=1179>`_ (1x)
* `Loopbio Triggerbox <http://loopbio.com/recording/>`_  with 5 *Trigger Ports* and 8 *State Ports* (1x): comes with Binder (3-pole) connectors for triggering cameras (5x), and Binder (6-pole) connectors for synchronization with external hardware (3x)

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/video_recording_hardware.png
   :width: 800
   :height: 320
   :align: center
   :alt: Video Hardware

.. raw:: html

   <br>

E-phys recording essentials
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `PXIe-1071 Chassis with power cord <https://www.ni.com/en-us/shop/model/pxie-1071.html>`_ (1x)
* `PXIe-8381 module and card with MXI-Express cable <https://www.ni.com/en-us/shop/model/pxie-8381.html>`_ (1x)
* `BNC-2110 <https://www.ni.com/en-us/support/model.bnc-2110.html>`_ (1x)
* `SHC68-68-EPM shielded cable <https://www.ni.com/en-us/support/model.shc68-68-epm.html>`_ (1x)
* `PXIe-6341 <https://www.ni.com/en-us/shop/model/pxie-6341.html>`_ (1x)
* `Neuropixels 2.0 probe <https://www.neuropixels.org/probe2-0>`_ (2x)
* `Neuropixels 2.0 headstage <https://www.neuropixels.org/probe2-0>`_ (1x)

.. figure:: https://raw.githubusercontent.com/bartulem/usv-playpen/refs/heads/main/docs/media/ephys_recording_hardware.png
   :width: 800
   :height: 320
   :align: center
   :alt: E-phys Hardware

.. raw:: html

   <br>

Other essentials
^^^^^^^^^^^^^^^^

* `Arduino Uno <https://store.arduino.cc/products/arduino-uno-rev3?srsltid=AfmBOoqCHxeme84k9_8zMTa3JTGYYzw20P36GEmJQBJGSvEcd48ShLBj>`_ (1x)
* `IR-LEDs <https://www.digikey.com/en/products/detail/marktech-optoelectronics/MTE9440M3A/2798891?so=88479393&content=productdetail_US&mkt_tok=MDI4LVNYSy01MDcAAAGVQEcEiS5xq-g7fZ0YNeAbQF1X6l1rQPO9OE8iU9Wud3fwZjjEL8KwezzzxWCu9NMbWbQtBvKalYDJcFjhdLc-2ckFNKIZoD6xJa_ac1xa>`_ (3x)
* `IR-reflectors <https://cmvisionsecurity.com/products/cmvision-cm-ir130-850nm-198pc-leds-300-400ft-long-range-ir-illuminator>`_ (4x)
* `3.0/4.0 mm IR-reflective markers <https://logemas.com/product/hemispherical-self-adhesive/>`_ (1x)
* `Raspberry Pi 4b <https://www.raspberrypi.com/products/raspberry-pi-4-model-b/>`_ (1x)
* `HiFiBerry DAC2 PRO <https://www.hifiberry.com/shop/boards/dac2-pro/>`_ (1x)
* `SA1 Stereo Amplifier <https://www.tdt.com/docs/hardware/sa1-stereo-amplifier/>`_ (1x)
* `ZB1PS Powered zBUS Device Chassis <https://www.tdt.com/docs/hardware/zb1ps-powered-zbus-device-chassis/>`_ (1x)
* `Sony MDREX15LP in-Ear Earbud Headphones <https://electronics.sony.com/audio/headphones/in-ear/p/mdrex15lp-b?srsltid=AfmBOopjpXrsT5eQPPYC-QkQGGfeTtJE50NBObAYFYOeHU5uB_7FvB03>`_ (1x)
* `3.5mm Female Jack to Bare Wire Open End TRS 3 Pole Stereo 1/8" 3.5mm <https://www.amazon.com/Fancasee-Replacement-Connector-Headphone-Earphone/dp/B07Y8LNMM6>`_ (1x)
* `BNC Male Balun Connector to 2 Screw Camera Terminal Male Adapter <https://www.amazon.com/Gagool-Connector-Terminal-Solderless-Surveillance/dp/B09DXVV5WV>`_ (1x)
* `Magnets for earbud mount <https://www.kjmagnetics.com/b222g-n52-neodymium-gold-plated-block-magnet>`_ (10x)
* `Intel Ethernet Converged X710-DA2 Network Adapter (X710DA2) for high speed ethernet <https://www.amazon.com/gp/product/B00NJ3ZC26/>`_ (2x)
* `Intel E10GSFPSR 10G SFP+ SR SFP for high speed ethernet <https://www.amazon.com/Intel-E10GSFPSR-10G-SFP-SR/dp/B016YK9CPI/>`_ (2x)


Software Requirements
---------------------

Audio PC essentials
^^^^^^^^^^^^^^^^^^^

Whatever operating system you are using, you will need to ensure the existence of the following software *prior to* using *usv-playpen*: (1) `git <https://git-scm.com/downloads>`_ (and add PATH to USER VARIABLES on Windows), (2) `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ (and add PATH to USER VARIABLES on Windows).

You can verify that the installation was successful by running the following commands in the terminal:

.. code-block:: bash

   git --version
   uv -version

`Avisoft Recorder USGH <https://avisoft.com/downloads/>`_ works on Windows 11. You should download a version of the software that does not require an USB license key. There is a known issue that the configuration file can occasionally contain invalid settings that lead to various unexpected behaviors. This potential issue can be avoided by disabling the automatic saving of the configuration when the program is closed (*Options > Configuration management > Save mode on exit > Save current configuration automatically*).
`CoolTerm <https://coolterm.en.lo4d.com/windows>`_ is a serial port terminal application, which allows you to record and keep Arduino print statements in the form of a text file. In the *usv-playpen/_config* directory,
you can find a CoolTerm configuration file, *coolterm_config.stc*, which you can import into CoolTerm. If you are setting CoolTerm up for the first time, it is best to unpack it in *D:\\CoolTerm* and further create two directories: *D:\\CoolTerm\\Connection_settings* (place the *coolterm_config.stc* file here) and *D:\\CoolTerm\\Data*. Open the config file in CoolTerm and ensure that the location of saved files is *D:\\CoolTerm\\Data* and not *D:\\CoolTerm*. An additional important point to consider is which port to set it to
(by default it is set to COM5). If you are using the existing Arduino UNO, it is sufficient to plug it into a port (ideally COM5, which would require no changes to the CoolTerm configuration file), as the sketch was already uploaded. However,
if you want to upload the sketch to a different Arduino device, you will need to install the `Arduino IDE <https://www.arduino.cc/en/software/>`_, compile and upload the following sketch: *usv-playpen/other/sychronization/generate_sync_pulses.ino*.

To control Ethernet connection status from the command line (more on this in the *Record* section), one needs to run Powershell in administrator mode. To ensure Powershell is in administrator mode all the time:
(1) find Windows PowerShell in windows search and pin it to task bar, (2) right-click on the icon in the task bar and right click again on Windows PowerShell, (3) in the Properties, go to
advanced properties and select *run as administrator* and hit OK. When you open PowerShell, it should say administrator in the title bar. To check if you are in administrator mode, type *whoami* and hit enter.
If you are in administrator mode, it should say *administrator*. It is also important to check the the file server is mounted to the PC. You can check all the mounted file systems with the following command:

.. code-block:: powershell

   gdr -PSProvider 'FileSystem'

If the file server is not mounted, you can mount it with the following command:

.. code-block:: powershell

   net use f: \\cup\falkner /user:username@princeton.edu password /persistent:yes
   net use m: \\cup\murthy /user:username@princeton.edu password /persistent:yes

One can also enable/disable the Ethernet connection with:

.. code-block:: powershell

   netsh interface set interface "ethernet_network_name" enable

Video PC essentials
^^^^^^^^^^^^^^^^^^^

There are two PCs running Ubuntu 18.04 LTS controlling camera acquisition with `Motif <http://loopbio.com/recording/>`_.
Three cameras are connected to the main PC, and two are connected to the secondary PC *via* the USB3 Micro-B Locking Cable. The main PC
is connected to the Loopbio Triggerbox and each camera is connected to the Triggerbox *via* the 6-pin GPIO cables. If necessary, one can
remote into each of the PCs using SSH. To do so, you need to know the password of the PC you want to connect to and have a stable VPN connection. You would connect in the following way:

.. code-block:: bash

   ssh labadmin@pni-<MAIN_PC_ID>.princeton.edu
   ssh labadmin@pni-<SECONDARY_PC_ID>.princeton.edu

If Motif is experiencing issues, it can be restarted on any PC with the following command:

.. code-block:: bash

   sudo systemctl restart supervisor.service

but the user then needs to connect to the **Motif web interface** and manually ensure that all the cameras are connected and ready for recording.

Another thing that needs to be ensured prior to recording is that the file server is mounted to the PC. You can mount the file server with the following command:

.. code-block:: bash

   //cup.princeton.edu/famousprof /mnt/famousprof cifs noauto,x-systemd.automount,credentials=/root/.credentials,iocharset=utf8,rw,file_mode=0664,dir_mode=0775,nolinux,noperm,vers=2.1 0 0

The video data is saved in /mnt/DATA of each computer.

EPHYS PC essentials
^^^^^^^^^^^^^^^^^^^

On the firmware/software side, install the following (note that SpikeGLX only works on Windows!):

* `SpikeGLX <https://billkarsh.github.io/SpikeGLX/>`_ (data acquisition software), unpack it in "C:\SpikeGLX"
* `PXI Enclustra Drivers <https://billkarsh.github.io/SpikeGLX/>`_ for your specific OS version (scroll down to locate the appropriate file)
* `NI Package Manager <https://www.ni.com/en/support/downloads/software-products/download.package-manager.html#322516>`_ (in case you ever get a NI-DAQ or some other NI device worth controlling)
* `NI-DAQmx <https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html#464560>`_

To make SpikeGLX functional (once the module and card are connected, and probe is connected to the chassis):

* put the probe configuration directories in "C:\SpikeGLX\Release_vXXXXXXXX-phaseXX\SpikeGLX\_Calibation"
* Download firmware for your specific SpikeGLX version, open the SpikeGLX Console and navigate to Tools > Update IMEC Firmware, select the slot you are using and:
* load BS firmware from, e.g., "C:\SpikeGLX\Release_vXXXXXXXX-phaseXX\Firmware"
* load BSC firmware from, e.g., "C:\SpikeGLX\Release_vXXXXXXXX-phaseXX\Firmware"

Setting up *usv-playpen*
^^^^^^^^^^^^^^^^^^^^^^^^

If you plan to conduct behavioral recordings, you need to modify */_config/motif_config.ini* to include the actual API key and SSH password to the tracking PCs:

.. code-block:: ini

   [motif]
   master_ip_address=10.241.1.205
   second_ip_address=10.241.1.183
   ssh_port=22
   ssh_username=labadmin
   ssh_password=XXX
   api=xxx

Likewise, if you plan to conduct behavioral recordings, you need to modify */_config/cup_config.ini* to include the actual username and password for the University Cup server:

.. code-block:: ini

    [cup]
    username=XXX
    password=XXX

If you plan to send/receive e-mail notifications when jobs start/complete, you need to modify */_config/email_config.ini*:

.. code-block:: ini

    [email]
    email_host=smtp.gmail.com
    email_port=465
    email_address=165b.pni@gmail.com
    email_password=xxx
