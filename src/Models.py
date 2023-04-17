import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, feature_length):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_length, 500),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(500, 100),
            nn.Dropout(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 500),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(500, feature_length)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded