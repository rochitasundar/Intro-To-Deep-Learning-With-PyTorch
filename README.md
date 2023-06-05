### DeepLearning-With-PyTorch

This repository contains my exercise code solutions to Udacity's coursework ['Intro to Deep Learning with PyTorch'](https://www.udacity.com/course/deep-learning-pytorch--ud188).

**Course 1: Introduction to Neural Networks**

  - Implementing a 1 layer neural network, training using gradient descent to predict student admissions at UCLA. 
  
**Course 2: Introduction to PyTorch** 

  - Implementing a multi layer neural network architecture using PyTorch for Fashion-MNIST dataset, using regularization to generalize performance on training and validation sets. 
  
**Course 3: Convolutional Neural Networks**

- Implementing a multi layer convolutional neural network architecture with convolutional, pooling and fully connected layers for CIFAR-10 dataset, using data augmentation and early stopping to improve performance. 
- Leveraging a pre-trained network: DenseNet, previously trained on ImageNet dataset and implementing transfer learning to classify pictures of cats and dogs.

**Course 4: Style Transfer**

- Leveraging a pre-trained network: VGG as a fixed feature extractor to create new images by merging the content of one image with the style of another image. The feature layers of a target image are compared with the original content and style image layers to calculate content loss and style loss (via., gram matrix calculations of convolutional layers) and used to iteratively update the target image via back propagartion. 
  
**Course 5: Recurrent Neural Networks**

- Implementing a 2-layer recurrent neural network architecture that predicts a next character when given an input text sequence. Text data is preprocessed, characters encoded as one hot encoded integer vectors and RNN trained to generate new text.

**Course 6: Sentiment Prediction RNNs**

- Implementing an embedding layer, 2 layer recurrent neural network architecture, fully connected layer with a sigmoid activation function to predict whether a movie review has a positive or negative sentiment. Text data is converted to lowercase, vectorized to integers, padded or truncated to maintain uniform input length, converted into tensors & batched as part of pre processing. A prediction function is written to carry out the necessary text pre-processing steps, & return an inference from the trained model.
