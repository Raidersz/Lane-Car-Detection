"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""
'''
import numpy as np
import matplotlib.image as mpimg
import cv2
from IPython.display import HTML
from IPython.core.display import Video
from moviepy.editor import *
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
from cardetection import *

class FindLaneLines:
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)
        
        if lane_detected:
            out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
            out_img = self.lanelines.plot(out_img)
            out_img = cardetection.detect(out_img)
        return out_img,lane_detected

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        #img = input_path
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, output_path):
        clip = VideoFileClip(0)
        #clip = input_path
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)
        
    def process_video_from_webcam(self, output_path):
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame == True:
                out_img, lane_detected = self.forward(frame)

            if lane_detected:
                out.write(out_img)

                # Show the processed frame in Jupyter Notebook
                clear_output(wait=True)
                display(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to stop capturing frames
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows() 

def main():
    findLaneLines = FindLaneLines()
    #clip1 = VideoFileClip("lanes_clip.mp4")
    #new_clip = clip1.set_fps(25)
    findLaneLines.process_video_from_webcam("outputwebcam.avi")

if __name__ == "__main__":
    main()

'''
import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from IPython.display import HTML
from IPython.core.display import Video
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
from cardetection import *

class FindLaneLines:
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        try:
           
           img = self.calibration.undistort(img)
           img = self.transform.forward(img)
           img = self.thresholding.forward(img)
           img = self.lanelines.forward(img)
           img = self.transform.backward(img)
       

           out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
           out_img = self.lanelines.plot(out_img)
           out_img = cardetection.detect(out_img)
        except :
            out_img = cardetection.detect(out_img)
        

        
        return out_img
    def resize_frame(frame,new_width,new_height):
        return cv2.resize(frame,(new_width,new_height),interpolation=cv2.INTER_AREA)

    def process_image(self,frame):
        frame = self.forward(frame)
        
        return frame
    
    def show_frame(self,frame):
       cv2.imshow('Processed Camera Feed', frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the video stream
        cv2.destroyAllWindows()
        return False
        
       return True

    '''def process_video(self, input_path, output_path, d):
        clip = VideoClip(input_path, duration = d)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False,fps = 20)
    import cv2
from moviepy.editor import VideoClip'''
    # def process_video(self, input_path, output_path):
    #     clip = VideoFileClip(input_path)
    #     out_clip = clip.fl_image(self.forward)
    #     out_clip.write_videofile(output_path, audio=False,fps = 20)
    import cv2
from moviepy.editor import VideoClip

# Function to capture frames from the laptop camera
def capture_frame(t):
    # Open the camera
    cap = cv2.VideoCapture(0)  # Change the parameter to 1 if using an external USB camera

    # Check if the camera is opened correctly
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Read a frame from the camera
    ret, frame = cap.read()

    # Release the camera
    cap.release()

    # If the frame was read successfully, return the frame as a numpy array
    if ret:
        return frame
    else:
        return None

# Function to convert numpy array to moviepy video frame
def make_frame(t):
    frame = capture_frame(t)
    return frame

def main():
     input_video_path = "challenge_video.mp4"
    
     findLaneLines = FindLaneLines()
     
     cap = cv2.VideoCapture(input_video_path)
     
     while cap.isOpened():
         ret,frame = cap.read()
         if not ret:
             break
         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
         #frame = findLaneLines.resize_frame(frame,1280,720)
         processed_frame = findLaneLines.process_image(frame)
         processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
         should_continue  = findLaneLines.show_frame(processed_frame)
     
         if not should_continue:
           break
     cap.release()  
     
     #findLaneLines.process_video("challenge_video.mp4","out1.mp4")

if __name__ == "__main__":
     main()

#clip =  VideoFileClip("out1.mp4")
#clip.preview()
'''while True:
    frame = capture_frame(0)
    if frame is not None:
        Fin  = FindLaneLines()
        processed_frame = Fin.process_image(frame)
        if not Fin.show_frame(processed_frame):
            break
    else:
        print("Error: Could not capture frame from camera.")
        ''break'''''