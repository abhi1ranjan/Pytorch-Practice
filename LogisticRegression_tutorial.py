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
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# prepare the dataset

breastCancer = datasets.load_breast_cancer()
X,y = breastCancer.data, breastCancer.target

n_samples, n_features = X.shape
print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale the feature - it is always recommended to scale the feature values when we are doing the logistic regression.
sc = StandardScaler()     # this gives 0 mean and unit variance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# convert to torch tensor from the numpy array.
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshape our test data
y_train = y_train.view(y_train.shape[0], 1)     # y has one row, we want to make it as a column vector, (each value in 1 row with only one column)
y_test = y_test.view(y_test.shape[0], 1)     # y has one row, we want to make it as a column vector, (each value in 1 row with only one column)


# 1. create the model
# In logistic regerssion, f = wx+b,and we apply sigmoid function at the end, for that we will have to make our own class.

class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()

        self.linear = nn.Linear(n_input_features, 1)    # we only have one layer, so output layer is 1

    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x))     # torch.sigmoid is built-in function, on that we apply the linear layer with x as data, return a value between 0 and 1
        return y_predicted

model = LogisticRegression(n_features)               # we have 30 features, sp self.linear will have 30 input features and one output, Hence our layer becomes 30*1 size

# 2. loss and optimizer
LearningRate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = LearningRate)

# 3. training loop
num_epochs = 200
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch +1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)     # sigmoid function return a value between 0 and 1, if y_predicted value is larger than 0.5, we will say it is class 1 else class 0.
    y_predicted_classes = y_predicted.round()

    accuracy = y_predicted_classes.eq(y_test).sum() / float(y_test.shape[0]) 
    print(f'accuracy = {accuracy:.4f}')