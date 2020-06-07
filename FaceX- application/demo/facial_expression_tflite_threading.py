# Seed value
seed_value= 0
 
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
 
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
 
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
 
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
 
# 5. Configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


import cv2 as cv
import datetime
import math
import time
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import dlib
import numpy as np
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return x - x_off, x + width + x_off, y - y_off, y + height + y_off

def get_color(emotion, prob):
    if emotion.lower() == 'angry':
        color = (0, 0, 255)
    elif emotion.lower() == 'disgust':
        color = (255, 0, 0)
    elif emotion.lower() == 'fear':
        color = (0, 255, 255)
    elif emotion.lower() == 'happy':
        color = (255, 255, 0)
    elif emotion.lower() == 'sad':
        color = (255, 255, 255)
    elif emotion.lower() == 'surprise':
        color = (255, 0, 255)
    else:
        color = (0, 255, 0)
    return color

def draw_bounding_box(image, coordinates, color):
    x, y, w, h = coordinates
    cv.rectangle(image, (x, y), (x + w, y + h), color, 3)
    
def draw_text(image, coordinates, text, color, x_offset=0, y_offset=0,
              font_scale=1, thickness=2):
    x, y = coordinates[:2]
    cv.putText(image, text, (x + x_offset, y + y_offset),
                cv.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv.LINE_AA)

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Let's make the EdgeCNN arhictecture based on DenseNet, but with some additional changes in order to work well on Raspberry Pi 4
import tensorflow as tf
from tensorflow.keras import layers, models, Model

# Create the function for Edge convolution block
# x: input layer 
# growthRate: how many filter we want fot output
# @return: the layer after applying convolution block 1 and 2
def EdgeBlock(x, growthRate, name):
  # Convolution block 1
  x = layers.Conv2D(4 * growthRate, kernel_size = 3, kernel_regularizer = tf.keras.regularizers.l1(0.0005),padding = 'same', name='conv1_dense' + name)(x)
  x = layers.BatchNormalization(name='batchnorm1_dense' + name)(x)
  x = layers.ReLU(name='relu_dense'+name)(x)

  # Convolution block 2
  x = layers.Conv2D(1 * growthRate, kernel_size = 3, kernel_regularizer = tf.keras.regularizers.l1(0.0005), padding = 'same', name='conv2_dense' + name)(x)
  x = layers.BatchNormalization(name='batchnorm2_dense' + name)(x)
  return x

# Create the function for Dense Block
# input_layer: the layer on which we apply the EdgeBlock function of 'repetition' times 
# growthRate: how many filter we want for output
# name: name of the denseblock densenumber_
# @return: concatenated layers after applying the EdgeBlock of 'repetition' times
def denseBlock(input_layer, repetition, growthRate, name):
  for i in range(repetition):
    # apply the convolution block 1 and 2
    x = EdgeBlock(input_layer, growthRate,str(name)+'_'+str(i))
    # concatenate with the input layer
    input_layer = layers.Concatenate()([input_layer,x])
  return input_layer

# Create the transition layer
# input_layer: the layer on which we apply average pooling
# name: name of the layer
# @return: the layer with average pooling applied
def transitionLayer(input_layer, name):
  input_layer = layers.AveragePooling2D((2,2), strides = 2, name = 'transition_'+str(name))(input_layer)
  return input_layer


# Function for creating the model 
def EdgeCNN(number_features, number_classes,growthRate):
    input_shape = (48,48,1)
    input_net = layers.Input(input_shape)
    # First layer of convolution
    x = layers.Conv2D(number_features,(3,3), kernel_regularizer = tf.keras.regularizers.l1(0.0005),padding = 'same', use_bias = True, strides = 1, name='conv0')(input_net)
    x = layers.MaxPool2D((3,3),padding = 'same',strides = 2, name='pool0')(x)

    # Add the Dense layers
    repetitions = 4, 4, 7

    layer_index = 1
    for repetition in repetitions:
        dense_block = denseBlock(x, repetition, growthRate,name=layer_index)
        x = transitionLayer(dense_block, name=layer_index)
        layer_index+=1

    x = layers.GlobalAveragePooling2D(data_format='channels_last')(dense_block)

    output_layer = layers.Dense(number_classes, activation='softmax', kernel_regularizer = tf.keras.regularizers.l1(0.0005))(x)

    return Model(input_net, output_layer)

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


detector = dlib.get_frontal_face_detector() # get frontal face detector from dlib
fps_vector = []
face_vetor = []

number_classes = 7
number_features = 32
growthRate = 8

# Path to tflite model
path = "models/facial_expression.tflite"
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# cap = cv.VideoCapture(0)
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

scaler = MinMaxScaler()
    
while(1):
    start = time.time()
    # get a frame
    frame = vs.read()
    height, width = frame.shape[:2]
    # show a frame
    frame = frame[100:100 + width, :]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    
    for face in faces:
        try:
            (x, y, w, h) = rect_to_bb(face)
            x1, x2, y1, y2 = apply_offsets((x, y, w, h), (10, 10))
            gray_face = gray[y1:y2, x1:x2]

            img = cv.resize(gray_face, (48, 48))
            img = scaler.fit_transform(img)
            img = img[np.newaxis,:, :, np.newaxis]
            img = img.astype('float32') 

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            predictions =  interpreter.get_tensor(output_details[0]['index'])
            
            emotion_index = predictions.argmax()
            emotion = class_names[emotion_index]

            prob = predictions[0][emotion_index]
            color = get_color(emotion, prob)
            
            text = emotion + ' ' + str(round(prob, 5) * 100)
            print(text)
            draw_bounding_box(image=frame, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color)
            draw_text(image=frame, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color, text=emotion)
        except Exception as e:
            print(e)  
    # update the FPS counter
    fps.update()
    cv.imshow("FACIAL EXPRESSION RECOGNITION VIDEO STREAM", frame) 
    if cv.waitKey(1) == ord('q'):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv.destroyAllWindows()
vs.stop()














