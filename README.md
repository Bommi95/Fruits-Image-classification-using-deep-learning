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
 
IMAGE AI result:
 ![alt text](https://github.com/Bommi95/Fruits-Image-classification-using-deep-learning/blob/master/apples_peaches1.jpg)


 Inpired by TensorFlow's object detection lab and  TF object detection API, later I started to use a hybrid model with FasterRCNN and InceptionV2, which yielded much better result than IMAGEAI. I ran the model on a TF-Hub trained module to perform object detection. 
 
 Loading module trained on TF hub:
  ```python
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]

detector = hub.load(module_handle).signatures['default']
```
 
 This pre-trained model is not 100% accurate, as it recognize a Pomegranate as an apple with 28% of chance(it did suspect the fruit is a pomergranate by detecting the correct class entity(pomergranate) but gave the entity a very low score). It is suggested to train the model with a set of fruits images only instead of using pretained weights to improve the prediction outcome. 

FasterRCNN+InceptionResNet V2 result:
![alt text](https://github.com/Bommi95/Fruits-Image-classification-using-deep-learning/blob/master/download%20(1).png)

Below is a list of all entities detected by the hybrid model. The output is taken from in Object Detection.ipynb file above.
```pythpn
l=('detection_class_entities','detection_scores')
for i in range(len(result['detection_class_entities'])):
    print({k:result[k][i] for k in l if k in result})
```
Output:
```pythpn
{'detection_class_entities': b'Apple', 'detection_scores': 0.8260584}
{'detection_class_entities': b'Apple', 'detection_scores': 0.72955704}
{'detection_class_entities': b'Flower', 'detection_scores': 0.5199092}
{'detection_class_entities': b'Fruit', 'detection_scores': 0.50979}
{'detection_class_entities': b'Flower', 'detection_scores': 0.37794763}
{'detection_class_entities': b'Flower', 'detection_scores': 0.3550301}
{'detection_class_entities': b'Apple', 'detection_scores': 0.30108768}
{'detection_class_entities': b'Apple', 'detection_scores': 0.2869468}
{'detection_class_entities': b'Fruit', 'detection_scores': 0.2133912}
{'detection_class_entities': b'Flower', 'detection_scores': 0.20451933}
{'detection_class_entities': b'Flower', 'detection_scores': 0.20076132}
{'detection_class_entities': b'Flower', 'detection_scores': 0.19108064}
{'detection_class_entities': b'Apple', 'detection_scores': 0.17397314}
{'detection_class_entities': b'Peach', 'detection_scores': 0.15915862}
{'detection_class_entities': b'Flower', 'detection_scores': 0.14678714}
{'detection_class_entities': b'Fruit', 'detection_scores': 0.14363769}
{'detection_class_entities': b'Apple', 'detection_scores': 0.14271991}
{'detection_class_entities': b'Rose', 'detection_scores': 0.0980814}
{'detection_class_entities': b'Flower', 'detection_scores': 0.08877069}
{'detection_class_entities': b'Flower', 'detection_scores': 0.08824175}
{'detection_class_entities': b'Food', 'detection_scores': 0.08261852}
{'detection_class_entities': b'Mango', 'detection_scores': 0.082389906}
{'detection_class_entities': b'Peach', 'detection_scores': 0.081454486}
{'detection_class_entities': b'Flower', 'detection_scores': 0.08067436}
{'detection_class_entities': b'Flower', 'detection_scores': 0.07243414}
{'detection_class_entities': b'Pear', 'detection_scores': 0.07195152}
{'detection_class_entities': b'Flower', 'detection_scores': 0.06842005}
{'detection_class_entities': b'Flower', 'detection_scores': 0.06635873}
{'detection_class_entities': b'Fruit', 'detection_scores': 0.065788396}
{'detection_class_entities': b'Vegetable', 'detection_scores': 0.062812686}
{'detection_class_entities': b'Apple', 'detection_scores': 0.06276963}
{'detection_class_entities': b'Food', 'detection_scores': 0.06251418}
{'detection_class_entities': b'Flower', 'detection_scores': 0.060398728}
{'detection_class_entities': b'Flower', 'detection_scores': 0.0598194}
{'detection_class_entities': b'Flower', 'detection_scores': 0.055693977}
{'detection_class_entities': b'Peach', 'detection_scores': 0.05357751}
{'detection_class_entities': b'Fruit', 'detection_scores': 0.052904032}
{'detection_class_entities': b'Flower', 'detection_scores': 0.04607338}
{'detection_class_entities': b'Tomato', 'detection_scores': 0.0452848}
{'detection_class_entities': b'Apple', 'detection_scores': 0.043812595}
{'detection_class_entities': b'Rose', 'detection_scores': 0.04303053}
{'detection_class_entities': b'Flower', 'detection_scores': 0.042907067}
{'detection_class_entities': b'Fruit', 'detection_scores': 0.042013388}
{'detection_class_entities': b'Flower', 'detection_scores': 0.037642106}
{'detection_class_entities': b'Peach', 'detection_scores': 0.037601557}
{'detection_class_entities': b'Peach', 'detection_scores': 0.037082803}
{'detection_class_entities': b'Vegetable', 'detection_scores': 0.030672248}
{'detection_class_entities': b'Flower', 'detection_scores': 0.029360486}
{'detection_class_entities': b'Peach', 'detection_scores': 0.029258657}
{'detection_class_entities': b'Flower', 'detection_scores': 0.028738065}
{'detection_class_entities': b'Peach', 'detection_scores': 0.02723589}
{'detection_class_entities': b'Lemon', 'detection_scores': 0.026796779}
{'detection_class_entities': b'Flower', 'detection_scores': 0.026136212}
{'detection_class_entities': b'Flower', 'detection_scores': 0.025140023}
{'detection_class_entities': b'Flower', 'detection_scores': 0.024559889}
{'detection_class_entities': b'Flower', 'detection_scores': 0.022777177}
{'detection_class_entities': b'Pomegranate', 'detection_scores': 0.022344902}
{'detection_class_entities': b'Rose', 'detection_scores': 0.022027092}
{'detection_class_entities': b'Food', 'detection_scores': 0.02186528}
{'detection_class_entities': b'Pomegranate', 'detection_scores': 0.021648038}
{'detection_class_entities': b'Flower', 'detection_scores': 0.021613903}
{'detection_class_entities': b'Flower', 'detection_scores': 0.020942777}
{'detection_class_entities': b'Flower', 'detection_scores': 0.020821538}
{'detection_class_entities': b'Flower', 'detection_scores': 0.02023027}
{'detection_class_entities': b'Fruit', 'detection_scores': 0.020029793}
{'detection_class_entities': b'Plant', 'detection_scores': 0.019550327}
{'detection_class_entities': b'Flower', 'detection_scores': 0.019500492}
{'detection_class_entities': b'Toy', 'detection_scores': 0.019018263}
{'detection_class_entities': b'Fruit', 'detection_scores': 0.018856008}
{'detection_class_entities': b'Orange', 'detection_scores': 0.018817624}
{'detection_class_entities': b'Plant', 'detection_scores': 0.017676858}
{'detection_class_entities': b'Flower', 'detection_scores': 0.017113473}
{'detection_class_entities': b'Fruit', 'detection_scores': 0.016921319}
{'detection_class_entities': b'Flower', 'detection_scores': 0.01685136}
{'detection_class_entities': b'Dessert', 'detection_scores': 0.016580293}
{'detection_class_entities': b'Flower', 'detection_scores': 0.016057262}
{'detection_class_entities': b'Tree', 'detection_scores': 0.015579125}
{'detection_class_entities': b'Tree', 'detection_scores': 0.015470242}
{'detection_class_entities': b'Tomato', 'detection_scores': 0.01521616}
{'detection_class_entities': b'Vegetable', 'detection_scores': 0.014478209}
{'detection_class_entities': b'Peach', 'detection_scores': 0.014129817}
{'detection_class_entities': b'Flower', 'detection_scores': 0.013901531}
{'detection_class_entities': b'Vegetable', 'detection_scores': 0.0138895465}
{'detection_class_entities': b'Pear', 'detection_scores': 0.01334956}
{'detection_class_entities': b'Flower', 'detection_scores': 0.0126789855}
{'detection_class_entities': b'Fruit', 'detection_scores': 0.012621822}
{'detection_class_entities': b'Clothing', 'detection_scores': 0.011986045}
{'detection_class_entities': b'Apple', 'detection_scores': 0.011861397}
{'detection_class_entities': b'Fruit', 'detection_scores': 0.011500125}
{'detection_class_entities': b'Flower', 'detection_scores': 0.0113948025}
{'detection_class_entities': b'Flower', 'detection_scores': 0.011327828}
{'detection_class_entities': b'Flower', 'detection_scores': 0.011199124}
{'detection_class_entities': b'Grape', 'detection_scores': 0.010932484}
{'detection_class_entities': b'Food', 'detection_scores': 0.010840527}
{'detection_class_entities': b'Orange', 'detection_scores': 0.010582318}
{'detection_class_entities': b'Grape', 'detection_scores': 0.010413458}
{'detection_class_entities': b'Rose', 'detection_scores': 0.009823482}
{'detection_class_entities': b'Flower', 'detection_scores': 0.009608548}
{'detection_class_entities': b'Flower', 'detection_scores': 0.009121277}
{'detection_class_entities': b'Lemon', 'detection_scores': 0.00904101}
```
## Reference
Classify images of clothing: 

https://www.tensorflow.org/tutorials/keras/classification 

Fruits 360: A dataset with 90483 images of 131 fruits and vegetables 
 
https://www.kaggle.com/moltean/fruit

IMAGE AI:
https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606

TF hub object detection:
https://www.tensorflow.org/hub/tutorials/object_detection
