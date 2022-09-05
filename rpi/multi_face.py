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
for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    start_time = time.time()
    # t1 = cv2.getTickCount()

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = np.copy(frame1.array)
    frame.setflags(write=1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = frame_gray
    faces = faceCascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
        )

    interpreter = tflite.Interpreter(model_path="tflites/" + args.model + ".tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    
    output_details = interpreter.get_output_details()
    count = 0

    for (x, y, w, h) in faces:
        count+=1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        box_size = max(h, w)
        roi_gray = frame_gray[y:y+box_size, x:x+box_size]
        roi_color = frame[y:y+box_size, x:x+box_size]
        face_gray = cv2.resize(roi_gray, (48,48))
        face_expanded = np.expand_dims(face_gray/255, axis=2).astype('float32')
        # Load the TFLite model and allocate tensors.

        interpreter.set_tensor(input_details[0]['index'], [face_expanded])
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        confidence = np.max(output_data[0]) * 100
        emo_state = np.where(output_data[0] == np.max(output_data[0]))[0][0]
        multi_eomtion_counter.append(emo_state)
        txt = str(mapper[emo_state])+"(%.02f%%)" % confidence
        cv2.putText(frame,txt, (x , y-10), font, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    
    # t2 = cv2.getTickCount()
    # time1 = (t2 - t1) / freq
    # frame_rate_calc = 1 / time1