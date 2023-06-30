import torch
import torch.nn as nn
import numpy as np

def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = Softmax(x)
print("Softmax with Numpy:", outputs)

y = torch.tensor([2.0, 1.0, 0.1])
Y_outputs = torch.softmax(y, dim=0)
print("Softmax with torch:", Y_outputs)

# =============================== Cross Entropy Loss ====================================#
# Cross-entropy loss, or log loss, measures the performance of a classification model 
# whose output is a probability value between 0 and 1. 
# -> loss increases as the predicted probability diverges from the actual label
def CrossEntropy(actual, predicted):
    EPS = 1e-15
    predicted = np.clip(predicted, EPS, 1 - EPS)
    loss = - np.sum(actual * np.log(predicted))
    return loss # / float (predicted.shape[0])   - this is to done to normalize the values

# Y - actual must be one-hot encoded
# if class 0: [1,0,0]
# if class 1: [0,1,0]
# if class 2: [0,0,1]

Y_actual = np.array([1, 0, 0])        # class is 0

# y_pred has probabilities
Y_pred_good = np.array([0.7, 0.2, 0.1])      # good prediction because it gives class 0 highest probability
Y_pred_bad = np.array([0.1, 0.3, 0.6])

loss1 = CrossEntropy(Y_actual, Y_pred_good)
loss2 = CrossEntropy(Y_actual, Y_pred_bad)
print(f'Loss1 Numpy: {loss1:.4f}')
print(f'Loss2 Numpy: {loss2:.4f}')

# CrossEntropyLoss in PyTorch (already applies Softmax)
# nn.LogSoftmax + nn.NLLLoss
# NLLLoss = negative log likelihood loss
# Y actual has class labels, no One-hot encoding
# Y_pred has raw scores(logits), no softmax
Loss_pytorch = nn.CrossEntropyLoss()

Y = torch.tensor([0])    # only class labels, no one-hot encoding

# size of y_pred =  nsamples * nclasses = 1*3
Y_pred1 = torch.tensor([[2.0, 1.0, 0.1]])     # these are the raw values, and class 0 has highest values - good prediction
Y_pred2 = torch.tensor([[0.5, 2.0, 0.3]])     # these are the raw values, and class 1 has highest values - bad prediction

loss1_pytorch = Loss_pytorch(Y_pred1, Y)
loss2_pytorch = Loss_pytorch(Y_pred2, Y)

print('Loss 1 Pytorch:',loss1_pytorch.item())
print('Loss 2 Pytorch:',loss2_pytorch.item())

_,prediction1 = torch.max(Y_pred1, 1)       # gives predictions which class have the highest probability
_,prediction2 = torch.max(Y_pred2, 1) 
# print(prediction1)
# print(prediction2)
print(f'Actual class: {Y.item()}, Y_pred1: {prediction1.item()}, Y_pred2: {prediction2.item()}')


# allows batch loss for multiple samples

# target is of size nBatch = 3
# each element has class label: 0, 1, or 2
Y = torch.tensor([2, 0, 1])

# input is of size nBatch x nClasses = 3 x 3
# Y_pred are logits (not softmax)
Y_pred_good = torch.tensor(
    [[0.1, 0.2, 3.9], # predict class 2
    [1.2, 0.1, 0.3], # predict class 0
    [0.3, 2.2, 0.2]]) # predict class 1

Y_pred_bad = torch.tensor(
    [[0.9, 0.2, 0.1],
    [0.1, 0.3, 1.5],
    [1.2, 0.2, 0.5]])

l1 = Loss_pytorch(Y_pred_good, Y)
l2 = Loss_pytorch(Y_pred_bad, Y)
print(f'Batch Loss1:  {l1.item():.4f}')
print(f'Batch Loss2: {l2.item():.4f}')

# get predictions
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y}, Y_pred1: {predictions1}, Y_pred2: {predictions2}')


# typical neural network with softmax function - doing binary classification
# Binary classification
# class NeuralNet1(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(NeuralNet1, self).__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size) 
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(hidden_size, 1)  
    
#     def forward(self, x):
#         out = self.linear1(x)
#         out = self.relu(out)
#         out = self.linear2(out)
#         # sigmoid at the end
#         y_pred = torch.sigmoid(out)
#         return y_pred

# model = NeuralNet1(input_size=28*28, hidden_size=5)
# criterion = nn.BCELoss()                                              # instead of using the softmax function, we use sigmoid function - Binary cross entropy loss

# Multiclass problem
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)                # 1st layer  - takes input size and gives output to the hidden layer
        self.relu = nn.ReLU()                                            # activation function in between
        self.linear2 = nn.Linear(hidden_size, num_classes)               # last layer is also linear, takes input from the hidden layer, and gives output for each class
    
    def forward(self, x):                                                # only apply our layer
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)       # create our model
criterion = nn.CrossEntropyLoss()  # (applies Softmax)
print('loss: ',criterion(Y_pred_good, Y))