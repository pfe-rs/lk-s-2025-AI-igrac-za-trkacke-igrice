import torch
import torch.nn as nn
import torch.nn.functional as F

class CarGameAgent(nn.Module):
    def __init__(self, input_size):
        super(CarGameAgent, self).__init__()

        num_actions = 4  # Gas, Steer Left, Steer Right, Brake

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Independent probabilities per action
        actions = torch.sigmoid(self.fc4(x))
        
        return actions
