"""
    We are going to replace the manually computed loss and parameter updates with PyTorch's built-in Loss and optimizer classes.
    Then we are goinf to replace the manually computed prediction model by implementing a pytorch model. 
    Then, Pytorch can do the complete pipeline for us.

    General Pipeline structure of a model:-
        1. Design model (input, output size, forward pass) - forward pass with all the different operations and different layers
        2. Construct loss and optimizer
        3. Training loop
            - forward pass: compute prediction
            - backward pass: gradients
            - update weights
    
    Returns:
        _type_: _description_
"""

import torch
import torch.nn as nn

print("=========================computing the gradient using the PyTorch==========================")
# f = w*x - this is the function we are trying to optimize  
# f = 2*x - Final function where W = 2

# l = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)   

# X = torch.tensor([1,2,3,4], dtype=torch.float32)   # we are creating a tensor of size 4 with values 1,2,3,4
# Y = torch.tensor([2,4,6,8], dtype=torch.float32)   # we are creating a tensor of size 4 with values 2,4,6,8

# W = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)      # We are initializing the weight to 0.0 and we are setting requires_grad=True so that pytorch can compute the gradients for us


# model prediction - manually computing the forward pass
# def forward(x):
#     return W * x
# we will replace the manual computation by implementing a pytorch model, we also don't need the weight parameter W anymore
# we are going to use the built in linear regression model from pytorch
# Now, we need to modify the input X and output Y to be 2D tensors, because pytorch expects the input to be 2D tensors
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)   # we are creating a tensor of size 4 with values 1,2,3,4
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)   # we are creating a tensor of size 4 with values 2,4,6,8

# model cannot have float values, so we are creating a float X_test tensor
X_test = torch.tensor([5], dtype=torch.float32)   # we are creating a tensor of size 4 with values 1,2,3,4

# we are getting the number of samples and number of features from the input X, we have 4 samples and for every sample we have 1 feature
N_samples, n_features = X.shape    
print(f'#samples: {N_samples}, #features: {n_features}')

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)   # we are creating a linear regression model with input size 1 and output size 1 - only have 1 layer. 

# let's create a custom model with multiple layers

class LinearRegression(nn.Module):       # nn.Module is the base class for all neural network modules, this class is getting derived from the nn.Module class
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()   # we are inheriting from the nn.Module class
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)   # we are creating a linear regression model with input size 1 and output size 1 - only have 1 layer. 

    def forward(self, x):
        return self.lin(x)
    

model = LinearRegression(input_size, output_size)   # we are creating a linear regression model with input size 1 and output size 1 - only have 1 layer.
 
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')   # we are passing the input X to the model and getting the output Y_pred


# loss function - manually computing the loss - MSE   - it is same syantax as numpy as in the pytorch also. 
# no need to define the loss function manually, we can use the built in loss function in pytorch
# def loss(y, y_predicted):
#     return ((y_predicted-y)**2).mean()

# Training
learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss()     # we are using the built in loss function in pytorch   - this a callable object, which takes 2 arguments, y and y_predicted and returns the loss value
# optimizer = torch.optim.SGD([W], lr=learning_rate)    # we are using the built in Stochastic gradient optimizer in pytorch
# also have to update the optimizer, as we don't have the weight parameter W anymore, replaced by model
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)    # we are using the built in Stochastic gradient optimizer in pytorch

for epoch in range(n_iters):
    # prediction = forward pass
    # y_pred = forward(X)
    y_pred = model(X)   # we are passing the input X to the model and getting the output Y_pred

    # loss
    l = loss(Y, y_pred)

    # gradients  = backward pass
    l.backward()    # dl/dw

    # update weights - this don't need to be part of computational graph, so we are using torch.no_grad() statement or using with statement
    # we don't need to manually update the weights, can use the built-in optimizer from pytorch
    # with torch.no_grad():
    #     W -= learning_rate * W.grad       # we are updating the weight W with the learning rate and the gradient dw in the negative direction because we want to minimize the loss
    
    optimizer.step()    # updating the weights

    # zero gradients after every epochs
    # W.grad.zero_()       #modifying inplace so we are using _ at the end of the function name.
    
    optimizer.zero_grad()   # we are using the built in optimizer from pytorch to zero the gradients

    if epoch % 5 == 0:
        [w, b] = model.parameters()   # we are getting the weight and bias from the model   - b is bias
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')


print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')   # we are passing the input X to the model and getting the output Y_pred