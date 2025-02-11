# Neural Network for Credit Scoring

This project is a simple neural network designed to predict whether a client should be approved for a loan based on input parameters.

## Architecture
Input Layer: accepts client parameters (age, job, marital, housing).
Hidden Layer: one hidden layer with Relu-activation function.
Output Layer: uses a sigmoid-activation function to classify the result as either loan approval or rejection.

## Training
The neural network is trained using gradient descent to minimize the mean squared error (MSE).
Gradients are calculated analytically using the backpropagation algorithm.

## Activation functions
Relu for the hidden layer.
Sigmoid for the output layer, converting the output to a probability between 0 and 1.

## Results
The neural network predicts loan approval decisions after training on the dataset.
