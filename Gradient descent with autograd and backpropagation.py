"""
    In this tutorial, we will be doing the following things:
    1. Prediction:- Manually computing the forward pass -  Pytorch Model will do this automatically
    2. Gradients Computation:- Manually computing the backward pass - pytorch Autograd computes the backward pass
    3. Loss Computation:- Manually computing the loss - Pytorch loss functions compute the loss  - just have to finalize the loss function
    4. Parameter Updates:- Manually updating the parameters  - Pytorch optimizers do this automatically - just have to finalize the optimizer
"""

# import torch
import numpy as np
print("=========================Manual computing the gradient using the NumPy==========================")
# f = w*x - this is the function we are trying to optimize
# f = 2*x - Final function where W = 2

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

W = 0.0      # We are initializing the weight to 0.0

# model prediction - manually computing the forward pass
def forward(x):
    return W * x

# loss function - manually computing the loss - MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()
"""
    MSE = 1/N * (w*x - y)**2
    dJ/dw = 1/N * 2x(w*x - y) - gradient computation    
"""
# gradient - manually computing the gradient
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()       # w*x is y_predicted

print(f'Prediction before training: f(5) = {forward(5):.3f}')


# Training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)   # gradient with respect to weight W which is our parameter

    # update weights
    W -= learning_rate * dw       # we are updating the weight W with the learning rate and the gradient dw in the negative direction because we want to minimize the loss

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {W:.3f}, loss = {l:.8f}')


print(f'Prediction after training: f(5) = {forward(5):.3f}')




#===================================================================================================#
# Now we will do the same thing using pytorch
import torch
print("=========================computing the gradient using the PyTorch==========================")
# f = w*x - this is the function we are trying to optimize  
# f = 2*x - Final function where W = 2

# l = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)   

X = torch.tensor([1,2,3,4], dtype=torch.float32)   # we are creating a tensor of size 4 with values 1,2,3,4
Y = torch.tensor([2,4,6,8], dtype=torch.float32)   # we are creating a tensor of size 4 with values 2,4,6,8

W = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)      # We are initializing the weight to 0.0 and we are setting requires_grad=True so that pytorch can compute the gradients for us


# model prediction - manually computing the forward pass
def forward(x):
    return W * x

# loss function - manually computing the loss - MSE   - it is same syantax as numpy as in the pytorch also.
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# Training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients  = backward pass
    l.backward()    # dl/dw

    # update weights - this don't need to be part of computational graph, so we are using torch.no_grad() statement or using with statement
    with torch.no_grad():
        W -= learning_rate * W.grad       # we are updating the weight W with the learning rate and the gradient dw in the negative direction because we want to minimize the loss
    
    # zero gradients after every epochs
    W.grad.zero_()       #modifying inplace so we are using _ at the end of the function name.

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {W:.3f}, loss = {l:.8f}')


print(f'Prediction after training: f(5) = {forward(5):.3f}')