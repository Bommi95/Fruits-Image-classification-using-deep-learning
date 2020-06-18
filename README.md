# Fruits-Image-classification-using-deep-learning

## Project Summary
 Fruit image classification and object detection is an important task in many business applications. A fruit classification system may be used to help a supermarket cashier identify the fruit species and prices. It may also be used to help people decide whether specific fruit species meet their dietary requirements. In this project, I propose an efficient framework for fruit classification using deep learning. More specifically, the framework is based on two different deep learning architectures. 

## Data Collection
 The image dataset come from Kaggle and contains 90483 images of 131 fruits and vegetables. I used CV2 and glob to build a pipeline to import the downloaded data and did training testing split.

## Model Building
 Keras sequential DNN model are used with 128 hidden layers and 5 output nodes(5 target labels: 'Apple Pink Lady','Banana','Mandarine','Limes','Peach')
Test accuracy reached 98.88%

## Result Visualization Sample
 I used pyplot to visualize the prediction result produced by a DNN using TensorFlow keras, also wrote a program to highlighted the image title and bar chart if the image is misclassified.

![alt text](https://github.com/Bommi95/Fruits-Image-classification-using-deep-learning/blob/master/test1.png)

## Object Detection Using FasterRCNN+InceptionResNet V2
 For my first approach, I used IMAGEAI, a python library for image recognition. I imported pretrained weights and implemented YOLOv3 (You Only Look Once) to do the object detection and draw bounding box around the object. 
 
 ![alt text](https://github.com/Bommi95/Fruits-Image-classification-using-deep-learning/blob/master/apples_peaches1.jpg)


 Inpired by TensorFlow's object detection lab and  TF object detection API, later I started to use a hybrid model with FasterRCNN and InceptionV2, which yielded much better result than IMAGEAI. I ran the model on a TF-Hub trained module to perform object detection. This pre-trained model is not 100% accurate, as it recognize a Pomegranate as an apple with 28% of chance(it did suspect the fruit is a pomergranate by detecting the correct class entity(pomergranate) but gave the entity a very low score). It is suggested to train the model with a set of fruits images only instead of using pretained weights to improve the prediction outcome. 
 
![alt text](https://github.com/Bommi95/Fruits-Image-classification-using-deep-learning/blob/master/download%20(1).png)

## Reference
Classify images of clothing: 

https://www.tensorflow.org/tutorials/keras/classification 

Fruits 360: A dataset with 90483 images of 131 fruits and vegetables 
 
https://www.kaggle.com/moltean/fruit

IMAGE AI:
https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606

TF hub object detection:
https://www.tensorflow.org/hub/tutorials/object_detection
