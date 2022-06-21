import cv2
import numpy as np
# 将工作目录更改到当前文件的上级目录下
import os


wp = os.path.abspath(os.path.dirname(__file__))
wp = os.path.abspath(os.path.join(wp, '..'))
os.chdir(wp)
import sys
sys.path.append(".")

import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

# print(os.getcwd())#显示当前路径
from face_toward import FaceTorward
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

face_t = FaceTorward(
        conf_path = "config/", 
        model_path = 'models',
        scene = 'non-mask',
        frame = frame,
        logger = logger
    )

while(1):
    # get a frame
    ret, frame = cap.read()
    frame = cv2.flip(frame,180)
    # show a frame
    face_t.detect(frame)
   
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break