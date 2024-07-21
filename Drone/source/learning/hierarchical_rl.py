import torch
import torch.nn as nn
import torch.nn.functional as F

class HighLevelPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(HighLevelPolicy, self).__init__()
        layers = []
        current_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class LowLevelPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(LowLevelPolicy, self).__init__()
        layers = []
        current_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class HierarchicalRLAgent:
    def __init__(self, high_level_policy, low_level_policy):
        self.high_level_policy = high_level_policy
        self.low_level_policy = low_level_policy

    def select_high_level_action(self, state):
        return self.high_level_policy(state)

    def select_low_level_action(self, state, high_level_action):
        combined_input = torch.cat((state, high_level_action), dim=1)
        return self.low_level_policy(combined_input)
