import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    def __init__(self, cnn_config):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn_layers = nn.ModuleList()
        in_channels = cnn_config.get('input_channels', 3)
        
        for i in range(1, 5):  # Assuming up to 4 conv layers
            conv_key = f'conv{i}'
            if conv_key in cnn_config:
                conv = cnn_config[conv_key]
                self.cnn_layers.append(nn.Conv2d(in_channels, conv['out_channels'], conv['kernel_size'], 
                                                 conv['stride'], padding=conv.get('padding', 0)))
                self.cnn_layers.append(nn.ReLU())
                in_channels = conv['out_channels']
        
        self.flatten = nn.Flatten()

    def forward(self, x):
        for layer in self.cnn_layers:
            x = layer(x)
        return self.flatten(x)

class PredictiveModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, cnn_config):
        super(PredictiveModel, self).__init__()
        self.cnn = CNNFeatureExtractor(cnn_config)
        self.cnn_output_size = self._calculate_cnn_output_size(cnn_config)
        
        # Define fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_size = input_size + self.cnn_output_size
        for hidden_size in hidden_layers:
            self.fc_layers.append(nn.Linear(prev_size, hidden_size))
            self.fc_layers.append(nn.ReLU())
            prev_size = hidden_size
        self.output_layer = nn.Linear(prev_size, output_size)

    def _calculate_cnn_output_size(self, cnn_config):
        dummy_input = torch.zeros(1, cnn_config.get('input_channels', 3), 
                                  cnn_config.get('image_height', 256), 
                                  cnn_config.get('image_width', 256))
        with torch.no_grad():
            dummy_output = self.cnn(dummy_input)
        return dummy_output.view(1, -1).size(1)

    def forward(self, state, action, image):
        cnn_output = self.cnn(image)
        x = torch.cat([state, action, cnn_output], dim=1)
        for layer in self.fc_layers:
            x = layer(x)
        return self.output_layer(x)

    def get_prediction(self, state, action, image):
        with torch.no_grad():
            return self.forward(state, action, image)

    def compute_loss(self, state, action, image, target):
        prediction = self.forward(state, action, image)
        return F.mse_loss(prediction, target)

    def to(self, device):
        return super(PredictiveModel, self).to(device)