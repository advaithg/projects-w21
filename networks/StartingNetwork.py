import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.FineTune import initialize_model


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, input_dim, output_dim, p_dropout):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x

class ConvNet(torch.nn.Module):
    def __init__(self, output_dim, network, p_dropout):  
        super().__init__()

        #pretrained transfer learning network:
        self.network, input_size = initialize_model(network, output_dim, False, use_pretrained=True)
        #self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.network = torch.nn.Sequential(*(list(self.network.children())[:-1]))
        #self.resnet.eval()
        self.first = StartingNetwork(512, output_dim, p_dropout)


    def forward(self, x):
        #with torch.no_grad():
        x = self.network(x)
        x = torch.squeeze(x)
        x = self.first.forward(x)
        return x