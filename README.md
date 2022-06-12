# CXR-Segmentation-Classification

This repository contains code to train a Machine Learning model to Segment CXRs for Lungs and Classify the Image to having 6 Diseases.

## CXR Segmentation

The Segmentation Model uses a U-NET architecture, where we use Double 2D Convolutions and MaxPool to reduce image resolution while increasing filters.
Although Batch Normalization was not introduced in the paper, we decided to add it since it increased our Validation Accuracy to 96%.

![alt text](https://drive.google.com/file/d/14blx-yvLAVoH4Bk-nhUXmlg4B1u-OiNb/view?usp=sharing "Prediction 01")
![alt text](https://drive.google.com/file/d/14lu6jUMoiOjpcwBqpqip_3wWrVp5EGMA/view?usp=sharing "Prediction 01")


## CXR Classification

For Classification, we tried may different models, to name a few:
* A standard few layers of convolution, followed by flattening everything out and ending with dense layers leading to prediction of each class
* A Model Inspired by the Research Paper [Pre-processing methods in chest X-ray image classification](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265949)
* Stand alone transfer learning using VGG16
* Stand alone transfer learning using ResNet50
* Stand alone transfer learning using MobileNetV2
* Stacked Transfer Learning Models (MobileNetV2 + DenseNet169)

After trying out the first model, we figured the training data wasnt enough to train the model. So, we opted for transfer learning. In transfer learning, the pretrained models were trained on a very different dataset (ImageNet). So, that didnot predict with a well enough accuracy.
We had to use images of resolution 224x224 at the beginning due to constraints put by Google CoLab. After not getting well enough results, we tried to change the Hyperparameters, increasing the batch size, increasing the resolution and epochs. These only seemed to increase the training time with little to no change in the accuracy. Higher number of epochs started overfitting the training data and increasing the training accuracy upto 70%.

### Further Work

After trying all of these, we came to a conclusion that we do not have enough data for training. We could potentially increase the dataset through data augmentation. By looking at the confusion matrix, we thought that we need much more data for EDEMA cases, and that our model is mispredicting ATELECTASIS, PLEURAL EFFUSION and CARDIOMEGALY. We could add another model at the end of this which would only be trained on these 3 cases and would help further classifying between these 3.

Another approach that could be useful would be to use transformers but they require a very large dataset. Given a large enough dataset, ViTs could help achieving a well enough accuracy.
