# NLP Author Classification (CNN)

Updated version: This project has been extended with model serving and web deployment.

Latest version: [ CNN Author Classification — TensorFlow Serving](https://github.com/curciojm/cnn_authorclass_serving)

Live demo: https://cnnauthorclassserving-production.up.railway.app/

This repository contains an earlier version of the project. The current version includes TensorFlow Serving, Docker, a Django web interface, and deployment on Railway.

This project implements a Convolutional Neural Network (CNN) for text classification, distinguishing between writing styles of two authors: Immanuel Kant and Sigmund Freud.

## Project Overview

The goal of this project is to classify long text excerpts based on author style using a deep learning approach. Although the task is relatively constrained due to the distinct writing styles of both authors, the main objective was to adapt and apply an existing NLP CNN architecture to a real dataset stored locally.

The original implementation was based on the book:
*Natural Language Processing with TensorFlow 2* by Ganegedara (2022).

## Model Architecture

The model is a CNN designed for text classification:

- Tokenization using Keras Tokenizer
- Word embeddings (64 dimensions)
- Multiple 1D convolutional layers (kernel sizes: 3, 4, 5)
- Concatenation of feature maps
- Global max pooling
- Dense softmax output layer

## Dataset

- The dataset consists of Spanish-language excerpts from translated works by Kant and Freud.
- Texts were extracted from digitized editions and preprocessed using a custom NLP pipeline.
- Labels: binary classification (Kant / Freud)

> Note: The raw dataset is not included due to copyright considerations.

## Preprocessing

- Text normalization (lowercasing, punctuation removal)
- Custom cleaning rules (domain-specific noise removal)
- Tokenization
- Sequence padding (based on 99th percentile length)

## Training

- Optimizer: Adam
- Loss: Sparse categorical crossentropy
- Batch size: 128
- Epochs: 5
- Callback: ReduceLROnPlateau

## Results

- Test accuracy: ~99.9%
- The model achieves near-perfect separation between authors due to highly distinctive writing styles.

## Reference

This project is an adaptation of the CNN architecture presented in:

Ganegedara (2022)  
*Natural Language Processing with TensorFlow 2*  
https://github.com/thushv89/packt_nlp_tensorflow_2

The original code was modified to work with a custom locally stored dataset.

