# FaceX - Edge Computing

FaceX is a `personal project` that allows you to get information about `people emotions` based on `their face`.

The purpose of this project is to show you a demo of how a system like this can be used. The project is intended to work on the edge devices like Raspberry Pi 4 and others.

## Table of contents
* [Demo](#demo)
* [General info](#general-info)
* [Requirements](#requirements)
* [Setup](#setup)
* [How to run](#how-to-run)
* [Features](#features)
* [Status](#status)
* [Inspiration](#inspiration)
* [Contact](#contact)

## Demo
<img src="https://github.com/pauldamsa/FaceX/blob/master/mysmile.png" height="500" width="500">

## General info
The purpose of this project, as I said above, is to work on edge devices. Due to the low computational power of such devices I had to find a suitable model in order to make the inference on the edge. The model used can be found [HERE](https://arxiv.org/pdf/1909.13522.pdf). Also, the conversion of the model in **TFLite** version was **mandatory** for running the inference on the edge. 

The dataset used for training the model is [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). Applying data augmentation was necessary for increasing the model performance.

In the **FaceX- research** directory you can find the jupyter notebooks which were used in the research environment. Also, you can find my analyse of FER2013 before and after data augmentation. In the [Training](https://github.com/pauldamsa/FaceX/blob/master/FaceX-%20research/Training/End-to-End%20Solution%20Facial%20expression%20recognition.ipynb) folder is the notebook for training. In the [FER2013 original and augmented analysis](https://github.com/pauldamsa/FaceX/tree/master/FaceX-%20research/analyzes/FER2013%20original%20and%20augmented%20analysis) folder are three notebooks where I did analysis of the dataset. The notebook from [model inference analysis](https://github.com/pauldamsa/FaceX/tree/master/FaceX-%20research/analyzes/model%20inference%20analysis) was used for analysing the performance of the model. Also, in the same package is the [performance tables](https://github.com/pauldamsa/FaceX/blob/master/FaceX-%20research/analyzes/model%20inference%20analysis/performance%20tables.html) file in which you can find the performance of the model. 

In the **FaceX- application** directory are stored the necessary packages for the application.

## Requirements
* TensorFlow - version 2.2.0
* OpenCV - version 4.1.2
* tensorboard - version 2.1.1
* scikit-learn - version 0.23.1
* matplotlib - version 3.2.1
* seaborn - version 0.10.1
* pandas - version 1.0.4
* plotly - version 4.8.1
* numpy - version 1.18.4
* imutils - version 0.5.3
* virtualenv - version 20.0.16

## Setup
The setup is simple, you need:
1. Create a virtual environment `python -m venv name_of_your_env`
2. Install the above packages with `pip install package_name`

## How to run
After you install the necessary packages you can run the app like this:
`python app.py`
The **app.py** file can be found in the **FaceX- application** directory.

## Features
List of features ready and TODOs for future development
* Based on the emotions frequency the app shows you a bar chart in order to get the most frequent emotion.

To-do list:
* Improve the performance of the model
* Get relevant data for training
* Get quality data 

###### The application doesn't have some complex features, because it wasn't the purpose.

## Status
Project is: _in progress_, because the purpose of the project was to get more knowledge about Edge Computing and ML tools. However, for more development on top of this project I think that this work is a good start point.

## Inspiration
The project was built for improving my **machine learning** and **data science skills**. Also, the inspiration was brought by a very good friend of mine who has a in-depth vision in this field. 

## Contact

If you want to contact me feel free to reach me at <paul_damsa9@yahoo.com>.
