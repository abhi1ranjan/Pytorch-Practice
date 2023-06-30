import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x+2

"""
-> First we calculate the forward pass of the network. We can see that y was created as a result of an operation, so it has the grad_fn attribute.
-> Requires_grad is set to True, so we can compute gradients with respect to x. Pytorch will create a function for us and that will help us in performing 
the backward pass.
-> Y has an attribute grad_fn. This attribute is an object that references a function that has created the Tensor. Function is add_backward, which will compute the 
gradient of y with respect to x in the backward pass.
"""

print(y)

z = y*y*2
z = z.mean()
print(z)


# Now we will compute the gradients with respect to x
z.backward()     # calculate the dz/dx, this creates a vector Jacobian product, i.e. dz/dx = [dz/dx1, dz/dx2, dz/dx3] to get the gradient of x
# we have jacobian matrix with parial derivatives of z with respect to x1, x2, x3, and then we multiply it with the gradient vector, 
# and then we will get the final gradients of z with respect to x.
print(x.grad)    # print the gradient of x


# now lets see what happens if we do not have requires_grad = True
a = torch.randn(3, requires_grad=False)
print(a)
b = (a+2)
print(b)
c = b*b*2
c = c.mean()
print(c)        # now we can see that c does not have grad_fn attribute, so we cannot compute the gradients with respect to a
# c.backward()   # this will give an error

# let's see what happens if we have more than one element in the tensor - don't apply mean
x = torch.randn(3, requires_grad=True)
print(x)
y = x+2
print(y)
z = y*y*2
print(z)
# z.backward()   # this will give an error, because we have more than one element in the tensor - grad can be implicitly computed only for scalar outputs.
# we have to create a vector of the same size as z with respect to which we want to compute the gradients
z.backward(torch.ones(3))   # this will give the same result as above, we can have a list of any 3 elements, and then we will get the gradients of z with respect to x.
print(x.grad)

# we should know how to prevent the gradients being tracked in some part of the code. 
# we have 3 methods to do that:
# 1. x.requires_grad_(False)
# 2. x.detach()    - this will return a new tensor with the same content as x, but with requires_grad set to False
# 3. with torch.no_grad():   - this will prevent tracking of gradients in the code block - wrap with 'with' statement
x.requires_grad_(False)
print(x)

x = torch.randn(3, requires_grad=True)
print(x)
y = x.detach()        # this will return a new tensor with the same content as x, but does not require gradients
print(y)

z = x+2
print(z)        # here z will have grad_fn attribute, because x has requires_grad = True

with torch.no_grad():
    z = x+2
    print(z)    # here z will not have grad_fn attribute, because we have wrapped it with torch.no_grad() statement


# Whenever we call the backward function, the gradient for this tensor will be accumulated in the dot_grad attribute in the instead of being replaced.
# Their values will be summed up. We can see this in the below example:
weights = torch.ones(4, requires_grad=True)
print(weights)
for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)     # here we can see that the gradients are being accumulated in the grad attribute
    weights.grad.zero_()    # this will reset the gradients to zero, so that we can accumulate the gradients again in the next epoch
    print(weights.grad)     # here we can see that the gradients are being reset to zero after calling the zero_() function


# lets see how optimizer works in pytorch with gradients
weights = torch.ones(4, requires_grad=True
optimizer = torch.optim.SGD(weights, lr=0.01)    # here we are using the SGD optimizer with learning rate of 0.01
optimizer.step()    # this will update the weights of the model - perfoms a single optimization step
print(weights)      # here we can see that the weights have been updated
optimizer.zero_grad()   # this will reset the gradients to zero, so that we can accumulate the gradients again in the next epoch

# Now let's see how to use the gradients to update the weights of the model
