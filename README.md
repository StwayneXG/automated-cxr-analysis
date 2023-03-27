# Project: Automated Chest X-ray Analysis

This repository contains the code to the end-term project for the advanced course of Digital Image Processing (EC-312) at College of Electrical & Mechanical Engineering (CEME), National University of Sciences and Technology (NUST), Rawalpindi, Pakistan. The project was supervised by Dr. Muhammad Usman Akram, Professor, CEME, NUST.

This project aims to develop an automated system for the analysis of chest X-rays (CXRs) to assist medical professionals in the diagnosis of different medical conditions. The project is divided in two parts:

* Identifying and segmenting the lungs in the image.
* Detecting the presence of various medical conditions (such as Atelectasis, Edema, Pleural Effusion, Consolidation, Cardiomegaly, or No Finding).

## Motivation

Close to a billion chest x-rays (CXRs) are taken around the globe every year for diagnosing different conditions that a patient may be suffering from. This large number of CXRs can lead to delays in the correct diagnosis and subsequent treatment as the medical professionals may not be able to keep up with such a large amount of data. Therefore, any kind of automation here can result in a system that is beneficial to both the medical professionals and the patients.

## Dataset

The dataset we used was provided by the [BioMedical Image and Signal Analysis (BIOMISA)](https://biomisa.org/) Research Group.

The segmentation dataset consists of 838 frontal-view CXR images. The dataset is divided into 2 parts:

* Training set: 762 images.
* Testing set: 76 images.

The classification dataset consists of 2003 frontal-view CXR images. The dataset is divided into 2 parts:

* Training set: 1,503 images.
* Testing set: 500 images.

The distribution of the classes in the training dataset is as follows:

<p align="center">
  <img src="https://iili.io/HNlFNIV.png" />
</p>

## Methodology
The project was divided into two parts:

The segmentation model was inspired by the "U-Net: Convolutional Networks for Biomedical Image Segmentation" paper [1]. The convolutional blocks consisted of two semi-blocks of a 3x3 convolutional layer (with zero padding) followed by a batch normalization layer and a ReLU activation function. An encoder block consisted of a convolutional block followed by a 2x2 max pooling layer. The decoder block consisted of a 2x2 transposed convolutional layer which was followed by a concatenation layer with the corresponding encoder block. The final output layer consisted of a 1x1 convolutional layer with a sigmoid activation function.

<p align="center">
  <img src="https://iili.io/HNc80aR.png" />
</p>

For the classification problem, we tried many different models:

* A standard convolution neural network, followed by a flattening layer and dense layers leading to final prediction of each class.
* A Model Inspired by the Research Paper "Pre-processing methods in chest X-ray image classification" [2].
* Stand alone transfer learning using VGG16.
* Stand alone transfer learning using ResNet50.
* Stand alone transfer learning using MobileNetV2.
* Stacked Transfer Learning Models (MobileNetV2 + DenseNet169).

Upon initial testing of the first model, we determined that the amount of training data provided was insufficient for effective model training. To overcome this challenge, we decided to explore transfer learning, where pre-trained models were utilized which were originally trained on a different dataset, specifically ImageNet. However, this approach did not yield satisfactory accuracy in our predictions.

## Results

For the segmentation task, we achieved a **Dice Coefficient of 0.96** on the test set. The following images show the results of the segmentation model on the test set:

Input Image            | Ground Truth           | Predicted Mask 
:-------------------------:|:-------------------------:|:-------------------------:
![Input Image](https://iili.io/HNcruUJ.png)  |  ![Ground Truth](https://iili.io/HNcrOW7.png)  |  ![Predicted Mask](https://iili.io/HNcrvfe.png)


For the classification task, we achieved an **accuracy of 0.39** on the test set. The following images show the results of the classification model on the test set:

Input Image            | True Label           | Predicted Label 
:-----:|:-------------------------:|:-------------------------:
<img src="https://iili.io/HNcsecX.jpg" width="256" height="256">  |  **Atelectasis**  |  **Atelectasis**
<img src="https://iili.io/HN1bDj1.md.jpg" width="256" height="256">  |  **Edema**  |  **Cardiomegaly**
<img src="https://iili.io/HN1pHx9.jpg" width="256" height="256">  |  **No Finding**  |  **No Finding**

Here is the confusion matrix for the classification task:

<p align="center">
  <img src="https://iili.io/HN1ZsYN.png" />
</p>

## Further Work

* The dataset provided was not sufficient for training a model that could achieve satisfactory results. Therefore, we plan to collect more data and train the model on that data.
* Data augmentation techniques can be used to artificially increase the size of the dataset.
* Ensemble learning can be used to improve the accuracy of the classification model.

## References

[1] Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation. In International Conference on Medical image computing and computer-assisted.](https://arxiv.org/abs/1505.04597) \
[2] Giełczyk, A., Marciniak A., Tarczewska, M., Lutowski, Z. (2022). [Pre-processing methods in chest X-ray image classification.](https://doi.org/10.1371/journal.pone.0265949)
