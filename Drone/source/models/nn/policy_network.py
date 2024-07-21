import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class CNNFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn_layers = nn.ModuleList()
        in_channels = config['image_channels']
        
        self.initial_conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.initial_relu = nn.ReLU()
        
        prev_out_channels = 32
        for i in range(1, 5):  # Up to 4 CNN layers
            if f'conv{i}' not in config['cnn']:
                break
            conv_config = config['cnn'][f'conv{i}']
            self.cnn_layers.append(nn.Conv2d(prev_out_channels, 
                                             conv_config['out_channels'], 
                                             conv_config['kernel_size'], 
                                             conv_config['stride']))
            self.cnn_layers.append(nn.ReLU())
            prev_out_channels = conv_config['out_channels']
        
        self.flatten = nn.Flatten()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        x = x.permute(0, 3, 1, 2)  # [batch, channels, height, width]
        x = self.initial_relu(self.initial_conv(x))
        
        for layer in self.cnn_layers:
            x = layer(x)
        
        return self.flatten(x)

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(attention_weights * x, dim=1)

class AdvancedPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, continuous, hidden_sizes, config):
        super(AdvancedPolicyNetwork, self).__init__()
        self.continuous = continuous
        self.cnn = CNNFeatureExtractor(config)
        
        # Calculate CNN output size
        dummy_input = torch.zeros(1, config['image_height'], config['image_width'], config['image_channels'])
        cnn_out = self.cnn(dummy_input)
        cnn_out_size = cnn_out.view(1, -1).size(1)
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU()
        )
        
        self.combined_size = hidden_sizes[0] + cnn_out_size
        self.attention = AttentionLayer(self.combined_size, hidden_sizes[1]) if config.get('use_attention', False) else None
        
        self.fc_layers = nn.ModuleList()
        in_size = self.combined_size
        for out_size in hidden_sizes[1:]:
            self.fc_layers.append(nn.Linear(in_size, out_size))
            if config.get('use_batch_norm', False):
                self.fc_layers.append(nn.GroupNorm(1, out_size))  # Use GroupNorm instead of BatchNorm
            self.fc_layers.append(nn.ReLU())
            if config.get('use_dropout', False):
                self.fc_layers.append(nn.Dropout(config.get('dropout_rate', 0.2)))
            in_size = out_size
        
        if continuous:
            self.mean = nn.Linear(hidden_sizes[-1], action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, state, visual):
        state_features = self.state_encoder(state)
        visual_features = self.cnn(visual)
        combined = torch.cat([state_features, visual_features], dim=1)
        
        if self.attention:
            x = self.attention(combined.unsqueeze(1)).squeeze(1)
        else:
            x = combined
        
        for layer in self.fc_layers:
            x = layer(x)
        
        if self.continuous:
            mean = self.mean(x)
            std = self.log_std.exp().expand_as(mean)
            return mean, std
        else:
            return F.softmax(self.action_head(x), dim=-1)
        
    def get_action(self, state, visual):
        with torch.no_grad():
            if self.continuous:
                mean, std = self.forward(state, visual)
                action_dist = Normal(mean, std)
                action = action_dist.sample()
                return action, action_dist.log_prob(action).sum(dim=-1)
            else:
                probs = self.forward(state, visual)
                action_dist = Categorical(probs)
                action = action_dist.sample()
                return action, action_dist.log_prob(action)

    def evaluate_actions(self, state, visual, action):
        if self.continuous:
            mean, std = self.forward(state, visual)
            action_dist = Normal(mean, std)
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            entropy = action_dist.entropy().mean()
        else:
            probs = self.forward(state, visual)
            action_dist = Categorical(probs)
            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy().mean()
        return log_prob, entropy

    def to(self, device):
        return super(AdvancedPolicyNetwork, self).to(device)