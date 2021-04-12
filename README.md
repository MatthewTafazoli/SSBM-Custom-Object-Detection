# SSBM-Custom-Object-Detection
Creating a custom object detection model for SSBM characters via Tensorflow 2.0's API


## About
Using [Tensorflow's 2.0 Custom Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md) to detect for characters in the game Super Smash Bros. Melee for the Nintendo Gamecube. 

The model up until this point has been trained/validated on tournament footage from Genesis 2 captured by AJPanton [youtube](https://www.youtube.com/channel/UCpPW_XqL3JszHcUpRunF-RA). The example footage below is taken from tournament footage from theWaffle77's [youtube](https://www.youtube.com/watch?v=BFyq-TGyMCc) channel from the original Genesis. 

For annotating the image set, I used [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/), exported the files as '.json files', and then used a script to convert the '.json' files to '.tfrec' for use in the 'pipline.config' file.

For those unfamiliar with Tensorflow's Object Detection API, I would recommend [this API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/). The link also contains information about using the built-in tensorboard features for monitoring your models training.

## Background and Further Refinements

The model was trained on a dataset involving 4 unique characters (Fox, Falco, Sheik, Peach), 2 stages and 1 color per character. There are 6 standard tournament stages, 26 characters and each character has between 4 to 6 separate costume colors. Currently the model generalizes decently to stages that haven't been trained on, however it generally does not do well on costume colors that are not in the dataset (ex. training on images of Red Fox does not currently help with identifying Blue Fox). The current limiting agents are mostly the computing power of my CPU, although I look to soon move to AWS or similar other options. I need to greatly expand the size of the dataset, which will take some time but there is essentially an unlimited amount of data available for this project. 

## Contents
`my_model/pipeline.config` - intializes the parameters for training the model

'my_model/saved_model/saved_model.pb' - The exported model after it has been trained on the imageset

`CharLabelMap.txt` - Label map file listing your classes; necessary for pipeline.config

'create_label_map.py' - Script to create label map

'JsonToTFrecord.py' - Takes the '.json' files output from the VGG Image Annotator and turns them into '.tfrec' files. Data is stored separately for each bounding box, so the .tfrec files grow quite large based on how many bounding boxes are used per image.

'evaluatemodel.py' - Run this script on an image folder to test the model on those images and save the resultant annotated images

'exporter_main_v2.py' - Official code from Tensorflow for exporting the model after completing training

'model_main_tf2.py' - Official code from Tensorflow for training the model on their Custom Object Detection

'testtfrec.py' - Tests the content of the '.tfrec' files you output from 

## Output Example
This gif is an example output of the model. The model works with one image at a time, and this is a compiling of about 3 seconds worth of 30 frame per seconds of gameplay.

![alt text](https://media.giphy.com/media/JuHdFmhfILqAgPLzY1/giphy.gif)
