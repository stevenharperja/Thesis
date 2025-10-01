import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module): #copy of prog 6 but in pytorch.
    def __init__(self, d=100, k=10):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(d, 2*d)         # input -> hidden1
        self.fc1 = nn.Linear(2*d, 2*d)       # hidden1 -> hidden2
        self.fc2 = nn.Linear(2*d, d)         # hidden2 -> hidden3
        self.fc3 = nn.Linear(d, k)           # hidden3 -> output

    def forward(self, x):
        x = torch.tanh(self.fc0(x))          # hidden1, tanh
        x = torch.tanh(self.fc1(x))          # hidden2, tanh
        x = F.relu(self.fc2(x))              # hidden3, relu
        x = torch.sigmoid(self.fc3(x))       # output, logistic (sigmoid)
        return x
