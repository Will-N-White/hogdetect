Provided in this github are the statistics from our model training and the weights we have aquired during testing. We have also included scripts for training with adjustable weights and formats, testing the model on a dataset, and running the object detection model through a camera. The only previous requirements are to download a dataset to train on and a separate dataset to perform tests on. 

#Disclosure
You can use the pretrained weights we have provided in our .zip files or train yourself. Darknet models are not yet functioning but may provide different results device to device. 

#How to Use (From Scratch)
1. Download preferred dataset in desired format and add paths to train.py.
2. Adjust parameters to match needs
3. Add paths to weights and test dataset in test.py
4. Perform image tests
5. Add weight path to camera.py
6. Set correct camera (0) for main webcam
7. Enjoy!!!

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
15. 




If you're looking to train or test a model through an .ipynb file, ArtemisHogTrainPC.ipynb has the latest scripts with train.py, test.py, and camera.py included. Also has the code to help install the necessary dependencies. Project Artemis is meant to help combat the growing issue of widespread devastation of cropland and help in the extermination of the invasive species wild pigs. This project will not and is not allowed to be used for any malicious intent. 
