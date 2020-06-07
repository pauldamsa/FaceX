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