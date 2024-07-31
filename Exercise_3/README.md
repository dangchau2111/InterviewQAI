# MNIST Triplet Loss Neural Network

## Introduction

This project uses a neural network to classify MNIST data using Triplet Loss. The main steps involved are:

1. Loading and preparing the MNIST dataset.
2. Building and training a simple neural network with one hidden layer and an output layer.
3. Using Triplet Loss to optimize the model.
4. Evaluating the model and predicting results for the test data.

## Formula Explanation

### Triplet Loss

Triplet Loss is a loss function designed to learn embeddings such that the distance between positive pairs (anchor and positive) is smaller than the distance between negative pairs (anchor and negative) by a certain margin.

The formula for Triplet Loss is:

$$ \[ \text{Loss} = \frac{1}{N} \sum_{i=1}^{N} \max\left(d_{\text{pos}}(i) - d_{\text{neg}}(i) + \text{margin}, 0\right) \] $$

Where:
- $\(d_{\text{pos}}(i)\)$ is the Euclidean distance between the anchor and positive.
- $\(d_{\text{neg}}(i)\)$ is the Euclidean distance between the anchor and negative.
- $\(\text{margin}\)$ is a fixed value to ensure the distance between the anchor and negative is greater than the distance between the anchor and positive.

### Gradient Calculation

During the backward propagation, gradients for weights and biases are computed to update them to minimize the loss. Specifically:
- **Gradient for weight `W2`**: Calculated by matrix multiplication of activation values of the hidden layer with the gradient of the loss.
- **Gradient for bias `b2`**: Calculated by summing up the gradient values and normalizing by the number of samples.

### Learning Rate Decay

Learning rate decay is applied to adjust the learning speed of the model over epochs:

$$ \[ \text{learning\_rate} = \frac{\text{initial\_learning\_rate}}{1 + \text{decay\_rate} \times \text{epoch}} \] $$

## Results

After training the model for 10 epochs with learning rate decay, the model achieved an accuracy on the test set of:
- Epoch 1/10, Loss: 0.9929696782140297, Learning Rate: 0.01
- Epoch 2/10, Loss: 0.9980345919063689, Learning Rate: 0.009990009990009992
- Epoch 3/10, Loss: 0.9988895028178308, Learning Rate: 0.00998003992015968
- Epoch 4/10, Loss: 0.9991689605712819, Learning Rate: 0.009970089730807579
- Epoch 5/10, Loss: 0.999327615965891, Learning Rate: 0.0099601593625498
- Epoch 6/10, Loss: 0.9994112410037328, Learning Rate: 0.009950248756218907
- Epoch 3/10, Loss: 0.9988895028178308, Learning Rate: 0.00998003992015968
- Epoch 4/10, Loss: 0.9991689605712819, Learning Rate: 0.009970089730807579
- Epoch 5/10, Loss: 0.999327615965891, Learning Rate: 0.0099601593625498
- Epoch 6/10, Loss: 0.9994112410037328, Learning Rate: 0.009950248756218907
- Epoch 5/10, Loss: 0.999327615965891, Learning Rate: 0.0099601593625498
- Epoch 6/10, Loss: 0.9994112410037328, Learning Rate: 0.009950248756218907
- Epoch 6/10, Loss: 0.9994112410037328, Learning Rate: 0.009950248756218907
- Epoch 7/10, Loss: 0.9994614077490388, Learning Rate: 0.009940357852882704
- Epoch 8/10, Loss: 0.9995121715861842, Learning Rate: 0.0099304865938431
- Epoch 9/10, Loss: 0.9995296268506397, Learning Rate: 0.00992063492063492
- Epoch 10/10, Loss: 0.9995738937151536, Learning Rate: 0.009910802775024779
- Accuracy: 72.16%

