"""
     General Pipeline structure of a model:-
        1. Design model (input, output size, forward pass) - forward pass with all the different operations and different layers
        2. Construct loss and optimizer
        3. Training loop
            - forward pass: compute prediction
            - backward pass: gradients
            - update weights
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# step 0 - prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)   # we are creating a dataset with 100 samples, 1 feature and noise=20
X = torch.from_numpy(X_numpy.astype(np.float32))   # we are converting the numpy array to a pytorch tensor
y = torch.from_numpy(y_numpy.astype(np.float32))   # we are converting the numpy array to a pytorch tensor
# we are reshaping the y tensor to be a 2D tensor -  because this y has 1 row only and we have to make a column vector out of it, by putting each value in one row
y = y.view(y.shape[0], 1)   # y.shape[0] is the number of rows in y, we are keeping the number of rows same and we are creating 1 column
n_samples, n_features = X.shape    # we are getting the number of samples and number of features from the input X, we have 100 samples and for every sample we have 1 feature
print(f'#samples: {n_samples}, #features: {n_features}')

# step 1 - model
input_size = n_features
output_size = 1
LinearModel = nn.Linear(input_size, output_size)   # we are creating a linear regression model with input size 1 and output size 1 - only have 1 layer.

# step 2 - loss and optimizer
learningRate = 0.01
criterion = nn.MSELoss()   # we are using the mean squared error loss function - a callable function
optimizer = torch.optim.SGD(LinearModel.parameters(), lr = learningRate)   # we are using the stochastic gradient descent optimizer - a callable function, we are passing the parameters of the model and the learning rate

# step 3 - training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = LinearModel(X)   # we are passing the input X to the model and we are getting the predicted output y
    loss = criterion(y_predicted, y)   # we are calculating the loss using the loss function

    # backward pass (gradient descent)
    loss.backward()   # we are computing the gradients


    # update weights
    optimizer.step()   # we are updating the weights of the model

    # zero gradients
    optimizer.zero_grad()   # we are setting the gradients to zero after every epoch

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


# plot
# detach create a tensor that shares storage with tensor that does not require grad, set gradient_requires =  False
predicted = LinearModel(X).detach().numpy()   # we are getting the predicted values from the model and we are converting it to a numpy array
plt.plot(X_numpy, y_numpy, 'ro')   # we are plotting the original data
plt.plot(X_numpy, predicted, 'b')   # we are plotting the predicted data
plt.show()   # we are showing the plot