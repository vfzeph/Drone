import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels, config):
        super().__init__()
        layers = []
        in_channels = input_channels
        
        for i in range(1, 5):
            conv_key = f'conv{i}'
            if conv_key in config:
                conv_config = config[conv_key]
                layers.extend([
                    nn.Conv2d(in_channels, conv_config['out_channels'], 
                              kernel_size=conv_config['kernel_size'], 
                              stride=conv_config['stride'],
                              padding=conv_config.get('padding', 0)),
                    nn.ReLU()
                ])
                in_channels = conv_config['out_channels']
            else:
                break

        self.layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.layers(x)
        return self.flatten(x)

class ICM(nn.Module):
    def __init__(self, icm_config):
        super(ICM, self).__init__()
        self.state_dim = icm_config['state_dim']
        self.action_dim = icm_config['action_dim']
        self.image_channels = icm_config['image_channels']
        self.image_height = icm_config['image_height']
        self.image_width = icm_config['image_width']

        self.cnn = CNNFeatureExtractor(self.image_channels, icm_config['cnn'])
        
        # Calculate CNN output size
        dummy_input = torch.zeros(1, self.image_channels, self.image_height, self.image_width)
        cnn_out = self.cnn(dummy_input)
        self.cnn_output_dim = cnn_out.view(1, -1).size(1)
        print(f"CNN output dimension: {self.cnn_output_dim}")

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, icm_config['state_encoder']['hidden_dim']),
            nn.ReLU()
        )

        forward_input_dim = self.cnn_output_dim + icm_config['state_encoder']['hidden_dim'] + self.action_dim
        print(f"Forward model input dimension: {forward_input_dim}")
        self.forward_model = nn.Sequential(
            nn.Linear(forward_input_dim, icm_config['forward_model']['hidden_dim']),
            nn.ReLU(),
            nn.Linear(icm_config['forward_model']['hidden_dim'], self.state_dim)
        )

        inverse_input_dim = self.cnn_output_dim * 2 + icm_config['state_encoder']['hidden_dim'] * 2
        self.inverse_model = nn.Sequential(
            nn.Linear(inverse_input_dim, icm_config['inverse_model']['hidden_dim']),
            nn.ReLU(),
            nn.Linear(icm_config['inverse_model']['hidden_dim'], self.action_dim)
        )

    def forward(self, state, next_state, action, image, next_image):
        state_feat = self.state_encoder(state)
        next_state_feat = self.state_encoder(next_state)
        image_feat = self.cnn(image)
        next_image_feat = self.cnn(next_image)

        combined_feat = torch.cat([state_feat, image_feat, next_state_feat, next_image_feat], dim=1)
        action_pred = self.inverse_model(combined_feat)
        
        forward_input = torch.cat([state_feat, image_feat, action], dim=1)
        next_state_pred = self.forward_model(forward_input)

        return state_feat, next_state_feat, action_pred, next_state_pred

    def intrinsic_reward(self, state, next_state, action, image, next_image):
        _, _, action_pred, next_state_pred = self.forward(state, next_state, action, image, next_image)
        
        forward_loss = F.mse_loss(next_state_pred, next_state, reduction='none').sum(dim=-1)
        inverse_loss = F.mse_loss(action_pred, action, reduction='none').sum(dim=-1)
        
        intrinsic_reward = forward_loss + inverse_loss
        return intrinsic_reward

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)

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

