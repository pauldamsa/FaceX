import cv2 as cv
from utils.utils import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Service:
    def __init__(self, net, face_detector):
        # the neural network
        self.net = net

        # the face detector
        self.face_detector = face_detector

        # the scaler for iamges
        self.scaler = MinMaxScaler()

        # init the emotion frequency vector
        self.emotion_frequency = [0,0,0,0,0,0,0]
    
    # Make the inference
    def inference(self, image_rgb):
        # make the image gray_scale
        gray_image = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)
        
        # get the faces
        faces = self.face_detector(gray_image, 1)

        # put the bounding boxes on the image
        annotated_image = self.set_bounding_boxes(gray_image,image_rgb,faces)
        return annotated_image
    
    def replace_values_of_emotion_index(self,lst):
        emotions = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}
        return [emotions.get(item, item) for item in lst]

    # Get the frequency
    def getRaport(self):
        lst = self.emotion_frequency
        lst.sort()
        lst_changed = self.replace_values_of_emotion_index(lst)
        return lst_changed

    # Set bounding boxes on image
    def set_bounding_boxes(self, gray_image, image_rgb, faces):
        # the labels
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        # iterate through faces
        for face in faces:
            try:
                (x, y, w, h) = rect_to_bb(face)
                x1, x2, y1, y2 = apply_offsets((x, y, w, h), (10, 10))
                gray_face = gray_image[y1:y2, x1:x2]

                img = cv.resize(gray_face, (48, 48))
                img = self.scaler.fit_transform(img)
                img = img[np.newaxis,:, :, np.newaxis]

                predictions = self.net.predict(img)
                
                emotion_index = predictions.argmax()
                self.emotion_frequency[emotion_index] += 1
                emotion = class_names[emotion_index]

                prob = predictions[0][emotion_index]
                color = get_color(emotion, prob)
                
                text = emotion + ' ' + str(round(prob, 5) * 100)
                print('[INFO] inference result ',text)
                image_rgb = draw_bounding_box(image=image_rgb, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color)
                image_rgb = draw_text(image=image_rgb, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color, text=emotion)
            except Exception as e:
                print(e)
        return image_rgb