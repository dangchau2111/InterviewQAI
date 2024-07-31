# Softmax Regression for MNIST Dataset

## Introduction

This repository contains an implementation of Softmax Regression for the MNIST dataset, a classic benchmark for handwritten digit classification. Softmax Regression is a generalized linear model for multiclass classification problems. In this implementation, we use NumPy to perform all necessary computations.

## Explanation

### Softmax Regression

Softmax Regression (or Multinomial Logistic Regression) is used for multiclass classification. It estimates the probability distribution over multiple classes given the input features.

**Softmax Function**:
The Softmax function converts logits (raw prediction values) into probabilities. It is defined as:

$$ P(y = k \mid \mathbf{x}; \mathbf{\theta}) = \frac{\exp(z_k)}{\sum_{j} \exp(z_j)} $$

where \( z_k \) is the logit for class \( k \) and the denominator sums over all possible classes \( j \).

**Loss Function**:
The Categorical Cross-Entropy Loss quantifies the difference between predicted probabilities and actual class labels. It is defined as:

$$ \text{Loss} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_{i,k} \log(p_{i,k}) $$

where:
- $\( m \)$ is the number of samples,
- $\( K \)$ is the number of classes,
- $\( y_{i,k} \)$ is a binary indicator (0 or 1) if class label $\( k \)$ is the correct classification for sample $\( i \)$,
- $\( p_{i,k} \)$ is the predicted probability of class $\( k \)$ for sample $\( i \)$.

**Gradient Descent**:
Gradient Descent is used to minimize the loss function by iteratively updating model parameters. The update rule is:

$$ \theta := \theta - \alpha \cdot \nabla_\theta \text{Loss} $$

where:
- $\( \alpha \)$ is the learning rate,
- $\( \nabla_\theta \text{Loss} \)$ is the gradient of the loss function with respect to $\( \theta \)$.

**Flow of Computation**:
1. Compute logits: $\( z = X \cdot \theta \)$
2. Apply Softmax to obtain class probabilities: $\( p = \text{softmax}(z) \)$
3. Compute Cross-Entropy Loss: $\( \text{Loss} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_{i,k} \log(p_{i,k}) \)$
4. Update parameters using Gradient Descent: $\( \theta := \theta - \alpha \cdot \nabla_\theta \text{Loss} \)$

## Execution

### Training Results

The model was trained for 1000 iterations with the following loss values recorded at various checkpoints:
- Iteration 0: Loss = 1.5888747638492926
- Iteration 100: Loss = 0.34419817129761715
- Iteration 200: Loss = 0.3092600330260409
- Iteration 300: Loss = 0.2938721864571801
- Iteration 400: Loss = 0.284682588933372
- Iteration 500: Loss = 0.27837018341530406
- Iteration 600: Loss = 0.27366197008872023
- Iteration 700: Loss = 0.26995334508320085
- Iteration 800: Loss = 0.26691662627460555
- Iteration 900: Loss = 0.26435794986878547

Accuracy: 91.89%


## References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Softmax Regression Explained]([https://towardsdatascience.com/softmax-regression-a-guide-for-machine-learning-4bdb4c30ff6c](https://machinelearningcoban.com/2017/02/17/softmax/#-gioi-thieu))

Feel free to explore the code and modify it to suit your needs!
