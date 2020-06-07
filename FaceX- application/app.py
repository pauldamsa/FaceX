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

from ui.ui import *
from service.service import Service
import dlib
from domain.EdgeCNN import *

# Create the network
number_classes = 7
number_features = 32
growthRate = 8

print('[INFO] loading the model...')
net = EdgeCNN(number_features, number_classes, growthRate) # get network architecture
net.load_weights('models/weights-improvement-299-0.65.hdf5') # get the weights of trained model

# Create the face_detector
print('[INFO] loading the face detector...')
face_detector = dlib.get_frontal_face_detector()

# Create the service
service = Service(net = net, face_detector = face_detector)

# start the app
print("[INFO] starting...")
pba = Application(service)
pba.start()