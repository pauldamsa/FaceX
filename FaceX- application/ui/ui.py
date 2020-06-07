import cv2 as cv
import os
from  service import *
from utils import *
import plotly.graph_objects as go

class Application:
    def __init__(self, service=None):
        """ Initialize application which uses OpenCV """
        self.vs = cv.VideoCapture(0) # capture video frames, 0 is your default video camera

        # Init service
        self.service = service

        # Start streaming
        self.video_loop()

    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter """
        ok, frame = self.vs.read()  # read frame from video stream
        if ok:  # frame captured without any errors
            annotated_frame = self.service.inference(frame) # get the annotated image 
            cv.imshow("FACIAL EXPRESSION RECOGNITION VIDEO STREAM", annotated_frame) # show every frame

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.vs.release()  # release web camera
        cv.destroyAllWindows() 
    
    def getRaport(self):
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        frequency = self.service.getRaport()

        # Use textposition='auto' for direct text
        fig = go.Figure(data=[go.Bar(
                    x=class_names, y=frequency,
                    text=frequency,
                    textposition='auto',
                )])

        fig.show()

    def start(self):
        while True:
            self.video_loop()
            if cv.waitKey(1) == ord('q'):
                self.destructor()
                self.getRaport()
                break