import cv2
import numpy as np

class CV2Manager:
    """Manages all CV2 drawing and window display"""
    def __init__(self, imW, imH, debug='none', preview_mode=True, window_name='Object detector'):
        self._debug = debug

        self.imW = imW
        self.imH = imH

        # Initialize frame rate calculation
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()

        self.window_name = window_name

        self.preview_mode = preview_mode
        if preview_mode:
            # Create window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Init variables used later in frame loop
        self.t1 = 0
        self.t2 = 0

    def convert_frame(self, frame, width, height):
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        frame_data = np.expand_dims(frame_resized, axis=0)

        return frame_data

    # box variable contains the 4 corners of bounding box
    def draw_bounding_box(self, frame, box, object_name, object_index, score):
        # Get bounding box coordinates and draw box
        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
        ymin = int(max(1,(box[0] * self.imH)))
        xmin = int(max(1,(box[1] * self.imW)))
        ymax = int(min(self.imH,(box[2] * self.imH)))
        xmax = int(min(self.imW,(box[3] * self.imW)))

        label = '%s: %d%%' % (object_name, int(score*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        
        color = (255,255,0) if object_name != 'person' else (0,0,255)
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
        
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        # Draw circle in center
        xcenter = xmin + (int(round((xmax - xmin) / 2)))
        ycenter = ymin + (int(round((ymax - ymin) / 2)))
        cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)

        if self._debug=='verbose':
            print('Object ' + str(object_index) + ': ' + object_name + ' at (' + str(xcenter) + ', ' + str(ycenter) + ') Probability: %' + str(score*100))

    def draw_fps(self, frame):
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(self.frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

    def display_window(self, frame):
        self.draw_fps(frame)
        if self.preview_mode:
            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow(self.window_name, frame)

    def calculate_fps(self):
        # Calculate framerate
        self.t2 = cv2.getTickCount()
        time1 = (self.t2-self.t1)/self.freq
        self.frame_rate_calc= 1/time1

    def start_frame_timer(self):
        # Start timer (for calculating frame rate)
        self.t1 = cv2.getTickCount()

    def cleanup(self):
        if self.preview_mode:
            # Clean up
            cv2.destroyAllWindows()
        print("CV2 cleanup success")
        
