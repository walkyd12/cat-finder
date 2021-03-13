import os
import numpy as np
import importlib.util

class Detector:
    """Object to perform object detection and classification on video frames"""
    def __init__(self, working_dir, model_name, graph_name, labelmap_name, use_TPU):
        # Set var for using a TPU
        self.use_TPU = use_TPU

        # If using Edge TPU, assign filename for Edge TPU model
        if self.use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (graph_name == 'detect.tflite'):
                graph_name = 'edgetpu.tflite'    

        # Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = os.path.join(working_dir,model_name,graph_name)
        # Path to label map file
        PATH_TO_LABELS = os.path.join(working_dir,model_name,labelmap_name)
        
        # Get list of labels for detection
        self.labels = self._get_labels(PATH_TO_LABELS)

        self.interpreter = self._get_tensor_interpreter(PATH_TO_CKPT)
        self.interpreter.allocate_tensors()

        # Gather model details from interpreter
        self.input_details, self.output_details = self._get_model_input_output_details()
        
        # Expand model details and set values
        self.input_std = 127.5
        self.input_mean = 127.5
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)


    # Open label text file and clean labels
    def _get_labels(self, label_path):
        # Load the label map
        with open(label_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if labels[0] == '???':
            del(labels[0])

        return labels

    # Import correct packages depending on tflite vs. tf and check for TPU packages
    def _get_tensor_interpreter(self, tf_graph_path):
        # Import TensorFlow libraries
        # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            if self.use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if self.use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if self.use_TPU:
            interpreter = Interpreter(model_path=tf_graph_path,
                                      experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(tf_graph_path)
        else:
            interpreter = Interpreter(model_path=tf_graph_path)

        return interpreter

    # Get model details from tflite interpreter
    def _get_model_input_output_details(self):
        # Get model details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        return input_details, output_details

    # Method to process a single frame from a video feed. Input Data comes from cv2 feed. Return classifications and bounding boxes
    def process_frame(self, input_data):
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
        self.interpreter.invoke()

        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects

        return boxes, classes, scores