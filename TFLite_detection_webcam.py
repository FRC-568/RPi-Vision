######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
import pickle
import struct
import zmq
import socket
import threading

#from utils import image_to_string

from threading import Thread
import importlib.util

import logging
from networktables import NetworkTables

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(120,120),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        self.stream.set(10, 0.000001)
        self.stream.set(11, 50)
        self.stream.set(12, 75)

        

        #video.set() commands
        # 0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
        # 1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
        # 2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
        # 3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
        # 4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
        # 5. CV_CAP_PROP_FPS Frame rate.
        # 6. CV_CAP_PROP_FOURCC 4-character code of codec.
        # 7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
        # 8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
        # 9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
        # 10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
        # 11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
        # 12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
        # 13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
        # 14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
        # 15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
        # 16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
        # 17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
        # 18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
        
def image_to_string(image):
    import cv2
    import base64
    encoded, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer)

def send_image_thread():
    
    # widthNumber = int(frame.shape[1] * scale_percent / 100)
    # heightNumber = int(frame.shape[0] * scale_percent / 100)
    # dim = (widthNumber, heightNumber)
    # # resize image
    # cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    
    image_as_string = image_to_string(frame)
    footage_socket.send(image_as_string)
    print("frame sent")

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'detect_edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
#with open(PATH_TO_LABELS, 'r') as f:
labels = ["shootingGoal", "robot", "ball"] #[line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)


#initialize networkTables
logging.basicConfig(level=logging.DEBUG)
NetworkTables.initialize(server="roborio-568-frc.local")
sd = NetworkTables.getTable("SmartDashboard")
goalCoords = NetworkTables.getTable("Goal Coordinates")
ballCoords = NetworkTables.getTable("Ball Coordinates")
otherRobots = NetworkTables.getTable("Robot Coordinates")
res = NetworkTables.getTable("Resolution")

#initialize camera streamer
server_address='10.5.68.187'
port='22'
print("Connecting to ", server_address, "at", port)
context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://' + server_address + ':' + port)
keep_running = True
 


#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index

            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            if object_name == "shootingGoal":
                
                widthOfBox = xmax - xmin
                heightOfBox = ymax - ymin

                centerXCoordinates = (xmin + xmax) / 2
                centerYCoordinates = (ymin + ymax) / 2
    
                goalCoords.putNumber("ymin", ymin)
                goalCoords.putNumber("ymax", ymax)
                goalCoords.putNumber("xmax", xmax)
        
                goalCoords.putNumber("centerX", centerXCoordinates)
                goalCoords.putNumber("centerY", centerXCoordinates)

                goalCoords.putNumber("boxWidth", widthOfBox)
                goalCoords.putNumber("boxHeight", heightOfBox)

                #Draw Bounding Box
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                # Draw label
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            if object_name == "ball":
                
                widthOfBox = xmax - xmin
                heightOfBox = ymax - ymin

                centerXCoordinates = (xmin + xmax) / 2
                centerYCoordinates = (ymin + ymax) / 2
    
                ballCoords.putNumber("ymin", ymin)
                ballCoords.putNumber("ymax", ymax)
                ballCoords.putNumber("xmax", xmax)
        
                ballCoords.putNumber("centerX", centerXCoordinates)
                ballCoords.putNumber("centerY", centerXCoordinates)

                ballCoords.putNumber("boxWidth", widthOfBox)
                ballCoords.putNumber("boxHeight", heightOfBox)

                #Draw Bounding Box
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                # Draw label
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
           
            if object_name == "robot":
                
                widthOfBox = xmax - xmin
                heightOfBox = ymax - ymin

                centerXCoordinates = (xmin + xmax) / 2
                centerYCoordinates = (ymin + ymax) / 2
    
                otherRobots.putNumber("ymin", ymin)
                otherRobots.putNumber("ymax", ymax)
                otherRobots.putNumber("xmax", xmax)
        
                otherRobots.putNumber("centerX", centerXCoordinates)
                otherRobots.putNumber("centerY", centerXCoordinates)

                otherRobots.putNumber("boxWidth", widthOfBox)
                otherRobots.putNumber("boxHeight", heightOfBox)

                #Draw Bounding Box
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                # Draw label
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
            
            
           

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    scale_percent = 60 # percent of original size
    
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    
    #sendFrame = videostream.read()
    #image_as_string = image_to_string(sendFrame)
    #footage_socket.send(image_as_string)
    #print("frame sent")
    cameraThread = threading.Thread(target=send_image_thread, args=())
    cameraThread.start()
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
