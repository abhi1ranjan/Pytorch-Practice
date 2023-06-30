'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda 

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

# dataset = torchvision.datasets.MNIST(
#     root = './data', transform = torchvision.transforms.ToTensor())         # here we apply ToTensor() transform, which will convert images or numpy arrays to tensor

# extend the dataset and dataloader tutorial lecture to be fit for transform operation.
# implement custom dataset

class WineDataset(Dataset):

    def __init__(self, transform = None):               # transform is optional, so default it is NONE.
        # data loading    
        data = np.loadtxt('/home/abhishekranjan/Documents/codes/pytorch tutorial/wine.csv', delimiter = ',', dtype = np.float32, skiprows = 1)
        
        # we don't need to convert the numpy array to torch tensor as that will be done by the transform operation, so leave as it is
        # self.x = torch.from_numpy(data[:, 1:])          # spliting the dataset into x and y. we want all the rows, but don't want 1st column. also convert it to torch.tensor
        # self.y = torch.from_numpy(data[:, [0]])         # size = [n_samples,1]
        self.x = data[:, 1:]
        self.y = data[:, [0]]

        self.n_samples = data.shape[0]                  # 1st dimension is total no of samples.

        self.transform = transform


    def __getitem__(self, index):                       # this will allow indexing
        # dataset[0]

        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def __len__(self):                                  # helps in finding the len of the dataset.
        # len(dataset)
        return self.n_samples
    

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

# implementing Multiply transform
class MulTransform:
    def __init__(self, factor):                         # we will need factorial, 
        self.factor = factor
    
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor

        return inputs, targets

dataset = WineDataset(transform = ToTensor())           #  if we put transform = None, we will get the data type of features and labels as numpy array only.
firstData = dataset[0]

# unpack the data into features and labels
features, labels =  firstData
print(type(features), type(labels))
print('features:- ',features, 'labels:- ', labels)


# lets see how composed transform work

composedTransform = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
NewDataSet = WineDataset(transform = composedTransform)
NewFirstData = NewDataSet[0]
Newfeatures, Newlabels =  NewFirstData
print('================= after using composed transform ===================')
print(type(Newfeatures), type(Newlabels))
print('features:- ',Newfeatures, 'labels:- ', Newlabels)
        
# # usage of the dataloader
# dataloader = DataLoader(dataset= dataset, batch_size = 4, shuffle = True, num_workers = 2)      # num_worker helps in faster loading of the data

# data_iter = iter(dataloader)
# Data = data_iter._next_data()
# feature, label = Data
# print('features found using DataLoader:- ',feature, 'labels found using DataLoader:- ', label)

# # we also iterate through the whole dataset
# # training loop
# num_epochs = 2
# total_samples = len(dataset)
# n_iterations = math.ceil(total_samples / 4)          # divided by 4 because batch size is 4
# print(total_samples, n_iterations)

# for epoch in range(num_epochs):                         # iterating over total no of epoch
#     for i, (inputs, labels) in enumerate(dataloader):   # iterating over the whole dataset, enumerate function will give the index and also the labels
#         # forward pass, backward pass and updates
#          if(i+1) % 5 == 0:
#              print(f'epoch {epoch + 1} / {num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')      
#              # it will print that there are 2 epoch, each epoch will have 45 iterations, printing data afte every 5th iterations, input size is 4*13, means we have 
#              # batch size equal to 4 and each batch have 13 features


## Pytorch also has many dataset laoded already like MNIST, fashion-mnist, cifar, coco etc
# torchvision.datasets.MNIST()
# torchvision.datasets.CIFAR10()
# torchvision.datasets.CocoCaptions()
