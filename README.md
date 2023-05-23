# 🧠 ASL Detection with Computer Vision and a CNN (Keras) 🧠
### The goal of the model is to use CV and a 3-layer (this may increase or decrese) CNN to successfully identify all 26 letters of the English alphabet from real time video capture. A user will be able to show their computer's webcam any ASL-baesd letter and the model will be able to output the corresponding symbol to the terminal.

## ⭐ Goals of Project:
    [x] Successfully run the project locally with acceptable performance
    [ ] Add functionality to store letters captured by webcam to stdout
    [x] Deploy the model
    [ ] Performance tuning
    
## Overview
This [video](https://www.youtube.com/watch?v=yqkISICHH-U&t=3083s) by Nicholas Renotte inspired me to build a project using object detection and a pretained model from the [Tensorflow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). After spending a ton of time trying to use the pretrained (on the famous COCO dataset) SSD MobileNet v2 320x320, its 640x640 variation, and even the SSD ResNet50 V1 FPN 640x640 (RetinaNet50). I was unsuccessful with these models. Each model failed to detect anything when more than 5 classes were added. No boundary boxes were even being drawn at first and then after retraining, too many boxes were drawn. 

I spent some time researching the pretrained Tensorflow SSD Mobilenet and noticed that a lot of people tried using them for the same project and were dealing with the same issues as me. I spent a few more days trying to fine tune the hyperparameters when I trained it on my own data. Still nothing. I refused to give up and started to research if using an LSTM would be the way to go. I tried installing the common Python package mediapipe and it would work (I couldn't use conda because I had run out of storage on my tiny 128gb MacBook). 

I decided to build a CNN with Keras. It's a 5 layer model (when only counting convolutional and dense layers), with batch normalization and dropout. These will help prevent overfitting, and increase the speed of training time. The training data was also binariazed, thus increasing the model's training time (this allowed it to converge faster). The shape of the model's input is (28, 28, 1).  

## Evaluating the Model 
<img src="https://github.com/oliviacarino/ASL_CNN_Project/blob/master/.readme_images/test_accuracy.png" width="800" height="500">

## References
1. [Deep Learning for Sign Language Recognition: Current Techniques, Benchmarks, and Open Issues](https://ieeexplore.ieee.org/document/9530569) by Al-Qurishi et al.
2. [Sign Language Recognition System using TensorFlow Object Detection API](https://arxiv.org/abs/2201.01486) by Srivastava et al.
3. I used the [Sign-Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) dataset from Kaggle.
4. This [Kaggle post](https://www.kaggle.com/code/madz2000/cnn-using-keras-100-accuracy/notebook) helped me get started on the ASL CNN detection project. I used the image preproessing steps, evaluation techniques and CNN model architecture.
5. Not super relevant to the entire project, but it helped me resolve an issue that I spent too many hours on and want to give credit. [link](https://stackoverflow.com/questions/59942348/cannot-reshape-array-of-size-2352-into-shape-1-28-28-1)
6. ___I plan on using some of the computer vision and hand segmentation information from [here](https://data-flair.training/blogs/sign-language-recognition-python-ml-opencv/) for real-time image capturing via a webcam.___
