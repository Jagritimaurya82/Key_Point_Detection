# T-shirt Key Point Detection using ResNet50

This repository contains code for training a neural network to detect key points on T-shirts. The model uses a ResNet50 architecture, pre-trained on ImageNet, and fine-tuned for this specific task. The dataset consists of images of T-shirts with annotated key points.

## Table of Contents
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)

  ## Dataset Preparation

To train the T-shirt key point detection model, you need to prepare a dataset that includes images of T-shirts and their corresponding key point annotations. Follow the steps below to prepare your dataset.

### Dataset Structure

The dataset should be structured as follows:

- A CSV file (`points_24.csv`) containing the file names of the images and their corresponding key point annotations.
- A directory containing the T-shirt images.

Ensure the dataset is organized like this:
      
    dataset/
    points_24.csv
    t-shirt_resized_train/
    image1.jpg
    image2.jpg
    ...


### Model Architecture

The model used for T-shirt key point detection is based on ResNet50, a powerful convolutional neural network originally designed for image classification. Here, it has been modified to output 24 key points for T-shirt images.

### ResNet50 Overview

ResNet50 is a deep residual network with 50 layers. It introduces skip connections or residual connections to allow the model to learn identity mappings, which helps in training very deep networks. ResNet50 is pre-trained on the ImageNet dataset and is fine-tuned for the specific task of detecting key points on T-shirts.

### Model Modification

To adapt ResNet50 for key point detection, the final fully connected layer is replaced to output 24 key points (48 values representing x and y coordinates for 24 key points).

### Code to Define the Model

Here's how you can define the modified ResNet50 model using PyTorch:

```python
import torch
import torch.nn as nn
import torchvision

# Load the pre-trained ResNet50 model
model = torchvision.models.resnet50(pretrained=True)

# Modify the final fully connected layer to output 24 key points (48 values)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=48)
