

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import math

# data = np.loadtxt('wine.csv')

# # training loop
# for epoch in range(1000):
#     x,y = data
# """
#     calculating gradient on large dataset is time consuming, so a better way is to calculate the gradient is to calculate in a batch size by dividing the data size
#     into small batches.    
#     """
# for epoch in range(total_batches):
#     # loop over all batches
#     x_batch,y_batch = ...
# # --> use DataSet and DataLoader to load wine.csv and pytorch will do the batch size dividing.

# """
#     epoch = 1 forward and backward pass of all training samples
#     batch_size = no of training samples in one forward and backward pass
#     number of iterations = number of passes, each pass using [batch_size] number of samples

#     e.g., 100 samples, batch_size = 20 --> 100/20 = 5 iterations for 1 epoch
#     """

"""
    There are 3 different types of class labels 1,2,3. 1st column is class label, rest all are feature values.
    """

# implement custom dataset
class WineDataset(Dataset):

    def __init__(self):
        # data loading    
        data = np.loadtxt('/home/abhishekranjan/Documents/codes/pytorch tutorial/wine.csv', delimiter = ',', dtype = np.float32, skiprows = 1)
        self.x = torch.from_numpy(data[:, 1:])          # spliting the dataset into x and y. we want all the rows, but don't want 1st column. also convert it to torch.tensor
        self.y = torch.from_numpy(data[:, [0]])         # size = [n_samples,1]

        self.n_samples = data.shape[0]                  # 1st dimension is total no of samples.


    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
dataset = WineDataset()
firstData = dataset[0]

# unpack the data into features and labels
features, labels =  firstData
print('features:- ',features, 'labels:- ', labels)
        
# usage of the dataloader
dataloader = DataLoader(dataset= dataset, batch_size = 4, shuffle = True, num_workers = 2)      # num_worker helps in faster loading of the data

data_iter = iter(dataloader)
Data = data_iter._next_data()
feature, label = Data
print('features found using DataLoader:- ',feature, 'labels found using DataLoader:- ', label)

# we also iterate through the whole dataset
# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)          # divided by 4 because batch size is 4
print(total_samples, n_iterations)

for epoch in range(num_epochs):                         # iterating over total no of epoch
    for i, (inputs, labels) in enumerate(dataloader):   # iterating over the whole dataset, enumerate function will give the index and also the labels
        # forward pass, backward pass and updates
         if(i+1) % 5 == 0:
             print(f'epoch {epoch + 1} / {num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')      
             # it will print that there are 2 epoch, each epoch will have 45 iterations, printing data afte every 5th iterations, input size is 4*13, means we have 
             # batch size equal to 4 and each batch have 13 features


## Pytorch also has many dataset laoded already like MNIST, fashion-mnist, cifar, coco etc
# torchvision.datasets.MNIST()
# torchvision.datasets.CIFAR10()
# torchvision.datasets.CocoCaptions()
