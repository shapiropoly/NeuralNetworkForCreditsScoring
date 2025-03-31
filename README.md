# Neural Network for Credit Scoring

This project is a simple neural network designed to predict whether a client should be approved for a loan based on input parameters.

At the moment, the network is working with an error: at some point during work, the parameter total_loss starts jumping up and down. I do not know why this is happening, and I am trying to find the error and fix it.

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
