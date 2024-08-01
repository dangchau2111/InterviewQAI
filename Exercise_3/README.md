# MNIST Triplet Loss Classification

## Introduction

This project involves classifying images from the MNIST dataset using a custom neural network model with triplet loss. The MNIST dataset consists of grayscale images of handwritten digits from 0 to 9, and this project aims to utilize deep learning techniques to accurately classify these digits.

## Workflow

1. **Data Loading and Preprocessing**:
   - The MNIST dataset is loaded using `fetch_openml` from `sklearn.datasets`.
   - The data is split into training and testing sets using `train_test_split`.
   - The features are normalized using `StandardScaler`.

2. **Model Definition**:
   - A neural network class (`NeuralNetwork`) is defined with an input layer, one hidden layer, and an output layer.
   - The network uses the `tanh` activation function for the hidden layer.

3. **Training Process**:
   - **Triplet Generation**: Triplets of anchor, positive, and negative samples are generated using the `create_triplets` method.
   - **Forward Pass**: The network performs a forward pass to compute embeddings for the anchor, positive, and negative samples.
   - **Loss Calculation**: The triplet loss is computed to measure how well the model is learning to differentiate between similar and dissimilar samples.
   - **Backward Pass**: The gradients are computed and weights are updated based on the computed loss.

4. **Evaluation**:
   - **Embedding Generation**: The model generates embeddings for both training and test data.
   - **Prediction**: The nearest neighbors in the embedding space are found using cosine distance to classify test samples.
   - **Accuracy Calculation**: The accuracy of the model on the test set is computed using `accuracy_score`.

## Mathematical Formulas

### Tanh Activation Function

The `tanh` activation function is defined as:

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

Where \( x \) is the input to the activation function. The `tanh` function maps input values to the range \([-1, 1]\), providing a non-linear transformation.

### Triplet Loss

The triplet loss function is used to ensure that the anchor sample is closer to positive samples (from the same class) than to negative samples (from different classes). The loss is computed as:

$$
\text{loss} = \frac{1}{N} \sum{i=1}^{N} \max(\text{posdist}i - \text{negdist}i + \text{margin}, 0)
$$

Where:
- $\( \text{posdist}i \)$ is the squared Euclidean distance between the anchor and positive embeddings for the \(i\)-th triplet.
- $\( \text{negdist}i \)$ is the squared Euclidean distance between the anchor and negative embeddings for the \(i\)-th triplet.
- $\( \text{margin} \)$ is a hyperparameter to ensure a minimum distance between positive and negative pairs.
- $\( N \)$ is the number of triplets.

The loss is averaged over all triplets to provide a single value for optimization.


### Gradient Calculation

To update the weights, the gradients are computed as follows:

1. **For Output Layer**:
   - Gradient of the weights and bias with respect to the loss is computed.

2. **For Hidden Layer**:
   - The gradient is propagated back using the chain rule, considering the derivative of the `tanh` function.

## Benefits of Deep Learning over Traditional Machine Learning Models

- **Feature Learning**: Deep learning models can automatically learn and extract features from raw data, whereas traditional models like softmax regression require manual feature engineering.
- **Handling Complex Patterns**: Deep learning models, particularly neural networks with multiple layers, can capture complex patterns and relationships in data that simple linear models may miss.
- **Higher Accuracy**: Deep learning models often achieve higher accuracy on large datasets due to their ability to model intricate patterns.
- **Scalability**: Deep learning models can scale with increased data and computational power, making them suitable for large-scale problems.

In summary, deep learning offers a significant advantage in tasks involving large and complex datasets by leveraging its ability to automatically learn and generalize features, outperforming traditional machine learning models in many scenarios.

