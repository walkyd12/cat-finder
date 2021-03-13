######## Webcam Object Detection Using Tensorflow-trained Classifier ################################
# Object detection inspired by article found at https://www.digikey.com/en/maker/projects/how-to-perform-object-detection-with-tensorflow-lite-on-raspberry-pi/b929e1519c7c43d5b2c6f89984883588
# Original Code Author: Evan Juras, 10/27/19; Edited by Shawn Hymel, 09/22/20
# original.py offers the source code that inspired this project
#####################################################################################################
# Author: Walker Schmidt
# Date: January 7th, 2021
# Description:
# This project uses a camera connected to the raspberry pi to stream a video feed through a
# tflite object detection model. This model returns bounding boxes, labels, and scores
# so that our output feed can draw frame by frame boxes on possbile objects. The goal is to detect
# a CAT on screen.
#
# Future interations will incorporate servos to move the camera when a CAT is detected, as well as
# play a sound in order to "scare" the cat. Ultimately, I don't want my cats on the kitchen counters. 
# Hoping this helps :)
######################################################################################################

# Import packages
import os
import argparse
import cv2
import time

from picamera.VideoStream import VideoStream
from picamera.CV2Manager import CV2Manager
from picamera.HighlightReel import HighlightReel

from object_detection.Detector import Detector

def init(args):
    # Pull arguments from args object to be used in initialization
    MODEL_NAME = args.modeldir # Tflite model file used for inference
    GRAPH_NAME = args.graph # Tflite graph file used for inference
    LABELMAP_NAME = args.labels # File containing possible labels for objects used in inference

    focus = args.focus # Object to focus on i.e. 'person' or 'cat' so that we can track a specific object
    resW, resH = args.resolution.split('x') # Get width and height of resolution
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu # Use coral TPU to offload inference to edge processing unit (speeds up inference > 5x)
    debug = args.debug # Debug statements that should be printed (none, basic, or verbose)

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Initialize tflite detector object for running inferences
    detector = Detector(CWD_PATH, MODEL_NAME, GRAPH_NAME, LABELMAP_NAME, use_TPU)

    # Initialize video stream
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)

    # Initialize CV2 Manager object to handle FPS count, drawing, etc.
    cm = CV2Manager(imW=imW, imH=imH, debug=debug, preview_mode=False, window_name=f'{focus} Finder')

    # Initialize Highlight reel object to help save important frames as well as create a video of all highlights
    reel = HighlightReel(CWD_PATH)

    # return objects to be used in main function
    return videostream, detector, cm, reel

def main(args, videostream, detector, cm, reel):
    focus = args.focus # Object to focus on for tracking
    min_conf_threshold = float(args.threshold) # Minimum threshold for successful object detection

    # Loop until user presses 'q' (Ctrl-C is also handled outside of this function)
    while True:

        # Start timer (for calculating frame rate)
        cm.start_frame_timer()

        # Grab frame from video stream
        frame_raw = videostream.read()

        # Create frame copy for use by CV2Manager
        frame = frame_raw.copy()

        # Resize frame and return frame data for use in model
        input_data = cm.convert_frame(frame, detector.width, detector.height)

        # Send frame data to tflite model to get infered detections including bounding box locations, labels, and scores
        boxes, classes, scores = detector.process_frame(input_data)

        # Save a list of all objects in the current frame. Init now, fill in the loop
        objects_in_frame = []

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get object name from labels and returned classes
                object_name = detector.labels[int(classes[i])] # Look up object name from "labels" array using class index
                objects_in_frame.append(object_name) # Add object_name to objects in frame

                # Draw bounding box for current object
                cm.draw_bounding_box(frame, boxes[i], object_name, i, scores[i])

        # Display the fame and the newly drawn bounding boxes/labels
        cm.display_window(frame)
        
        # Check if this frame includes what we are looking for. If so, save the frame to the reel
        if focus in objects_in_frame:
            reel.save_frame(frame)

        # Use fps and timer to calculate the updated frames per second value
        cm.calculate_fps()

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

def cleanup(cm, videostream, reel):
    # Cleanup our highlight reel, cv2 manager, and stop the videostream
    reel.cleanup()
    cm.cleanup()
    videostream.stop()

if __name__=="__main__":
    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--focus', help='Label to focus on. Defaults to "person"',
                        default='person')
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=True)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')
    parser.add_argument('--debug', help='Print debug information. none=no debug print statements, basic=minumum prints, verbose=maximum prints',
                        default='none')
                        
    args = parser.parse_args()

    try:
        videostream, detector, cm, reel = init(args)
    except Exception as e:
        print(f'Error initializing, exiting application . {e}')
    else:
        try:
            main(args, videostream, detector, cm, reel)
        except KeyboardInterrupt:
            print(f'Keyboard interupt, exiting and performing cleanup.')
            cleanup(cm, videostream, reel)
        else:
            # Exit of program happened cleanly, should still cleanup
            cleanup(cm, videostream, reel)

    print("Exiting...")
