import cv2
import numpy as np
import argparse
from picamera.array import PiRGBArray
from picamera import PiCamera
import tflite_runtime.interpreter as tflite
import time
from DoSomething import DoSomething
import json
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model name', default = 'overall_best')
args = parser.parse_args()

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up camera constants
IM_WIDTH = 1280//4
IM_HEIGHT = 192 #720//4

# Initialize frame rate calculation
# frame_rate_calc = 1
# freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# translate model output to label
mapper = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

camera = PiCamera()
camera.resolution = (IM_WIDTH, IM_HEIGHT)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
rawCapture.truncate(0)

test = DoSomething("publisher 2")
test.run()
multi_counter = 0
multi_eomtion_counter = []
one_face_fps = []
two_face_fps = []
three_face_fps = []
one_face = False
two_face = False
three_face = False
state_counter = 0
non_equal_counter = 0
previous_state = 0
one_time_sender = True
fixed_previous = -20
emo_state = 0
multi_time = time.time()