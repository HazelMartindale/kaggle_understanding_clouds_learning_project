# kaggle Understanding Clouds Learning Project 

Work on the kaggle competition understanding clouds from satellite images to practice skills of image based deep learning.
https://www.kaggle.com/c/understanding_cloud_organization/overview 

This project is being used as a learning project for deep learning and image segmentaiton. 
The data from the kaggle project is not hosted in this repo and should be downloaded seperatly.
The notebook should work standalone but has only been tested on Colab. 
No attempt has been made to test the code on other platforms. 

The notebook to date uses 3 versions of U-net architecture with different pretrained weights. 
Further refinement of the method will likely improve results. 

The code files contain the supporting functions to create masks from encoded pixels, define the data generator, and the model architecure.
These were moved into seperate folders to make the notebook easier to follow.

