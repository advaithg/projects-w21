import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.FineTune import initialize_model


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))        
        x = (self.fc2(x))
        return x

class ConvNet(torch.nn.Module):
    def __init__(self, output_dim, network):  
        super().__init__()

        #pretrained transfer learning network:
        self.network, input_size = initialize_model(network, output_dim, False, use_pretrained=True)
        #self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.network = torch.nn.Sequential(*(list(self.network.children())[:-1]))
        #self.resnet.eval()
        
        self.first = StartingNetwork(512, output_dim)


    def forward(self, x):
        #with torch.no_grad():
        x = self.network(x)
        x = torch.squeeze(x)
        x = self.first.forward(x)
        return x