Provided in this github are the statistics from our model training and the weights we have aquired during testing. We have also included scripts for training with adjustable weights and formats, testing the model on a dataset, and running the object detection model through a camera. The only previous requirements are to download a dataset to train on and a separate dataset to perform tests on. 

#Disclosure
You can use the pretrained weights we have provided in our .zip files or train yourself. Darknet models are not yet functioning but may provide different results device to device. 
#Datasets
Datasets for all models can be found by the following links
https://app.roboflow.com/artemis-xexhs/original-rkwr4/1
https://app.roboflow.com/artemis-xexhs/white-hot-nightvision/1
https://app.roboflow.com/artemis-xexhs/rainbow-piuzl/1
https://app.roboflow.com/artemis-xexhs/inferno-vqxa8/1
https://app.roboflow.com/artemis-xexhs/black-hot-3o0iw/1

#How to Use (From Scratch)
1. Download preferred dataset in desired format and add paths to train.py.
2. Adjust parameters to match needs
3. Add paths to weights and test dataset in test.py
4. Perform image tests
5. Add weight path to camera.py
6. Set correct camera (0) for main webcam
7. Enjoy!!!

#Use with Raspberry Pi5 on Pi OS
1. Create a virtual environment
2. Download and ensure all depencies listed below are installed
3. Use FFplay command from ffmpeg to determine which port the cameras are connected to
4. Adjust Artemis.py to ensure camera ports are correctly configured
5. Change directory to project directory
6. python Artemis.py (Python version 3.12 reccommended)
7. This should run. If not make sure all file pathings to models and ports to cameras are correct and try again. 

#Libraries needed
1. cv2 (OpenCV)
2. playsound
3. time
4. numpy (np)
5. YOLO
6. ultralytics
7. random
8. matplotlib.pyplot as plt
9. os
10. argparse
11. itertools
12. torch
13. yaml
14. pygame
15. ffmpeg




If you're looking to train or test a model through an .ipynb file or through terminal with a .py, ArtemisHogTrainPC.ipynb has the latest scripts with train.py, test.py, and camera.py included. Also it has the code to help install the necessary dependencies. ArtemisTrainFinal.py is a training script that will run in a virtual environment using all datasets provided listed above from roboflow running through them iteratively to help determine better accuracy and precision parameters for each dataset. Project Artemis is meant to help combat the growing issue of widespread devastation of cropland and help in the management of the invasive species wild pigs. This project will not and is not allowed to be used for any malicious intent. 
