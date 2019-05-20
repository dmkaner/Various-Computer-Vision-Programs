import numpy as np 
import torch
import matplotlib.pyplot as plt 
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

# plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
# plt.show()

# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.fc1 = nn.Linear(784, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         x = F.softmax(x, dim=1)

#         return x
# model = Network()

input_size = 784
hidden_sizes = [400, 200, 100]
output_size = 10

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_sizes[0])),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
    ('relu3', nn.ReLU()),
    ('output', nn.Linear(hidden_sizes[2], output_size)),
    ('softmax', nn.Softmax(dim=1))])
)
