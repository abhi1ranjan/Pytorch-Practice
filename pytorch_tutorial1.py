## we will learning about the tensors and how to use them in pytorch
## also learn how to convert numpy array to pytorch tensor and vice versa
## we will also learn about the autograd feature of pytorch

import torch
import numpy as np

x = torch.empty(1)          # empty tensor of size 1
print(x)

x = torch.empty(3)          # empty tensor of size 3  (1D tensor or vector with 3 elements)
print(x)

x = torch.empty(2, 3)       # empty tensor of size 2x3 (2D tensor or matrix with 2 rows and 3 columns)
print(x)

x = torch.empty(2, 3, 4)    # empty tensor of size 2x3x4 (3D tensor or matrix with 2 rows, 3 columns and 4 depth)
print(x)

x = torch.empty(2, 3, 4, 5) # empty tensor of size 2x3x4x5 (4D tensor or matrix with 2 rows, 3 columns, 4 depth and 5 height)
# print(x)                    ## it is hard to visualize 4D tensor

x = torch.rand(2, 3)        # random tensor of size 2x3 (2D tensor or matrix with 2 rows and 3 columns)
print(x)


x = torch.zeros(2, 3)           # tensor of zeros of size 2x3 (2D tensor or matrix with 2 rows and 3 columns)
print(x)

x = torch.ones(2, 3)            # tensor of ones of size 2x3 (2D tensor or matrix with 2 rows and 3 columns)
print(x)
print(x.dtype)

x = torch.ones(2, 3, dtype=torch.int)            # tensor of ones of size 2x3 (2D tensor or matrix with 2 rows and 3 columns)
print(x)


# we can also create tensors from python lists or datasets
x = torch.tensor([2.5, 0.1])    # tensor of size 2x1 (2D tensor or matrix with 2 rows and 1 column)
print(x, x.dtype)

# performing operations on tensors
x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
z = x + y
print(z)
z = torch.add(x, y)
print(z)

"""
    In Pytorch, any function that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.
    Those function will always perform in-place operation.
"""

# we can also do inplace addition
y.add_(x)
print(y)

# we can also do element wise substraction, inplace subtraction, multiplication, inplace multiplication, division, inplace division
z = x - y
print(z)
z = torch.sub(x, y)
print(z)
y.sub_(x)
print(y)

z = x * y
print(z)
z = torch.mul(x, y)
print(z)
y.mul_(x)
print(y)

z = x / y
print(z)
z = torch.div(x, y)
print(z)

# we can also do slicing operation on tensors just like numpy
x = torch.rand(5, 3)
print(x)
print(x[:, 0])      # all rows and 0th column
print(x[1, :])      # 1st row and all columns
print(x[1, 1])      # 1st row and 1st column
print(x[1, 1].item())   # 1st row and 1st column value - item() is used to get the value of a single element tensor

# we can also do reshaping of tensors
x = torch.rand(4, 4)
print(x)
y = x.view(16)      # reshaping 4x4 tensor to 16x1 tensor - view() is used to reshape the tensor - 1d tensor
print(y)
# reshaping 4x4 tensor to 2x8 tensor - view() is used to reshape the tensor - 2d tensor - -1 means infer the value of that dimension automatically based on other dimensions
y = x.view(-1, 8)   
print(y)
print(y.shape)
print(y.size())

# converting numpy array to pytorch tensor and vice versa

a = np.ones(5)
print(a, type(a))
b = torch.from_numpy(a)     # converting numpy array to pytorch tensor
print(b, type(b))

a = torch.ones(5)
print(a, type(a))
b = a.numpy()               # converting pytorch tensor to numpy array
print(b, type(b))


# we have to careful of where the tensor is stored - cpu or gpu, then both numpy and pytorch tensor will share the same memory location 
# and changing one will change the other, so we have to be careful of this. 
a.add_(1)                  # changing pytorch tensor, add 1 to all elements
print(a)
print(b)                    # numpy array also changed - it is not changed in my case because numpy is on cpu and pytorch tensor is on gpu.
# If we are changing the numpy array, then pytorch tensor will also change
c = np.ones(10)
print(c)
d = torch.from_numpy(c)     # converting numpy array to pytorch tensor
print(d)
c += 1                   # changing numpy array, add 1 to all elements
print(c)
print(d)                    # pytorch tensor also changed

# check if cuda is available or not
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# moving tensors to gpu
x = torch.ones(5, device=device)
print(x)
y = torch.rand(5)
y = y.to(device)       # moving tensor to gpu, we can also do y = y.cuda()
print(y)
z = x + y
print("addition operation",z)              # adding tensors on gpu, we can also do z = x.cuda() + y.cuda()

# moving tensors back to cpu
# before moving tensors to cpu, if we try to convert it to numpy array, it will give error because numpy array can only be created from cpu tensors
# z.numpy()               # this will give error
# so first we have to move the tensor to cpu and then convert it to numpy array
z = z.to("cpu")         # moving tensor to cpu, we can also do z = z.cpu()
print(z)
print(z.numpy())        # converting tensor to numpy array



# autograd feature of pytorch
# autograd is a feature of pytorch that automatically does the backpropagation for us
# we have to set the requires_grad=True for the tensors for which we want to calculate the gradients
# we can also set the requires_grad=True for the tensors which are created by performing operations on other tensors
xTrue = torch.ones(2, 2, requires_grad=True)     # setting requires_grad=True for the tensor
print(xTrue)
