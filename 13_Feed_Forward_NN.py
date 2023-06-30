"""
    MNIST Dataset
    DataLoader, Transformation
    Multilayer Neural Net, Activation Function
    Loss and Optimizer
    Training Loop (batch training)
    Model evaluation
    GPU support

    This will do the digit classification based on the MNIST dataset
"""

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'CPU')
print(device)

# hyperparameters
input = 784             # image size is 28*28 and we will flatten this in 1-d array or tensors
HiddenSize = 100        # try different values here
NumClasses = 10         # we have 10 different classes in the dataset - from 0 to 9
NoOfEpochs = 2
BatchSize = 100
learningRate = 0.01

# import dataset
TrainDataset = torchvision.datasets.MNIST(root='./data', train= True, transform=transforms.ToTensor(), download=True)
TestDataset = torchvision.datasets.MNIST(root='./data', train= False, transform=transforms.ToTensor())

TrainLoader = torch.utils.data.DataLoader(dataset=TrainDataset, batch_size=BatchSize, shuffle=True)

TestLoader = torch.utils.data.DataLoader(dataset=TestDataset, batch_size=BatchSize, shuffle=False)

# check the dataset
examples = iter(TrainLoader)
# unpack the sample
samples, label = examples._next_data()
# torch.Size([100, 1, 28, 28]) -> batch size = 100, channel = 1, 28*28 is image array; torch.Size([100]) - for each class label we have one value here
print(samples.shape, label.shape)         


# plot the data
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')            # samples[i][0] -> gives 1st row and 1st channel data

# plt.show()


# Now we want to classify these dataset
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):     # her num_classes is the output size
        super (NeuralNet, self).__init__()
        self.linearLayer1 = nn.Linear(input_size, hidden_size)    # input size is input_size, and output size is hidden_size
        self.relu = nn.ReLU()
        self.linearLayer2 = nn.Linear(hidden_size, num_classes)   # input size is hidden_size, and output size is num_classes
        # we haven't applied any activation function because we don't need that as contrast to the multi-class classification problems
        # we will apply the cross-entropy function as a loss function, which would apply the softmax function, so nothing to apply here.

    def forward(self, x):
        out = self.linearLayer1(x)
        out = self.relu(out)
        out = self.linearLayer2(out)
        return out
    

ClassfiyModel = NeuralNet(input, HiddenSize, NumClasses).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ClassfiyModel.parameters(), lr = learningRate)

# training loop
NoTotalSteps = len(TrainLoader)                                   # total no of steps is equal to the data points present in the training loader
for epoch in range(NoOfEpochs):
    for i, (images, labels) in enumerate(TrainLoader):            # unpacking the trainloader will give us the index, and data in the form of tuple - images and labels
        # reshape the image, currently it is 100, 1, 28, 28 
        # our input size is 784. so, our tensor need size of 100, 784 - 2d tensor
        images = images.reshape(-1, 28*28).to(device)             # -1 will find the batch size automatically, pushing it to the GPU using "to(device)" function
        labels = labels.to(device)


        # forward pass
        PredOutputs = ClassfiyModel(images)
        loss = criterion(PredOutputs, labels)                     # calculate the loss by passing the predicted output and actual labels

        # backward pass
        optimizer.zero_grad()                                     # empty the gradient inorder to not to track in the computational graph
        loss.backward()                                           # Perform the backpropagation
        optimizer.step()                                          # will do the update step and updates the parameters of the NN

        if (i+1) % 100  == 0:
            print(f'epoch {epoch +1} / {NoOfEpochs}, Current step {i+1}/{NoTotalSteps}, loss = {loss.item():.4f}')


# Testing part
# here we don't want to compute the gradient for all the steps we do, So wrap this up 
with torch.no_grad():
    NoCorrectPred = 0
    NoSamples = 0
    # loop over all the batches in the samples
    for images, labels in TestLoader:
        # reshape the images
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # calculate the prediction
        outputs = ClassfiyModel(images)

        # get actual prediction - torch.max() return the value and index of the image
        # we interested in the index, this will give us the class labels
        _, predictions = torch.max(outputs, 1)
        NoSamples += labels.shape[0]                                    # this gives us the no of samples in the current batch - should be 100
        NoCorrectPred += (predictions == labels).sum().item()            # for each correct predictions, we will add +1

    accuracy = 100.0 * NoCorrectPred / NoSamples
    print(f'accuracy = {accuracy}')


