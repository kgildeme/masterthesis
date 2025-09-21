import torch 
import torch.nn as nn
import torch.nn.functional as F

from .modules import Conv2d_ReLU, ResidualBlock2D, Conv1D_ReLU, ResidualBlock1D, ConvTranspose1D_ReLU

class Autoencoder(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):    
        return self.encoder(x)  
    
    def decode(self, x):    
        return self.decoder(x)


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims, activation=F.relu, dropout=None, bias=True, activation_out=False):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dims[0], bias=bias)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1], bias=bias) for i in range(len(hidden_dims)-1)])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        self.activation_out = activation_out

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        if self.activation_out:
            return self.activation(self.output_layer(x))
        return self.output_layer(x)

class CNN_1D(nn.Module):

    def __init__(self, feature_dims, residual_dims: list = None, kernel_size=3, flatten_and_ff: int =None, window=None):
        super().__init__()

        self.layer_1d = nn.ModuleList([
            Conv1D_ReLU(
                feature_dims[i],
                feature_dims[i+1],
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2
            ) for i in range(len(feature_dims)-1)
        ])

        self.layer_res = None
        if residual_dims is not None:
            self.layer_res = nn.ModuleList([
                ResidualBlock1D(
                    feature_dims[-1], residual_dims[i]
                ) for i in range(len(residual_dims))
            ])

        self.output = []
        if flatten_and_ff is not None:
            self.ouptut = nn.ModuleList([
                nn.Flatten(),
                nn.Linear(feature_dims[-1]*window, flatten_and_ff)
            ])
    
    def forward(self, x):
        for l in self.layer_1d:
            x = l(x)
        for l in self.layer_res:
            x = l(x)
        for l in self.output:
            x = l(x)
        return x
   
        
class CNNTranspose_1D(nn.Module):

    def __init__(self, feature_dims, residual_dims: list = None, kernel_size=3, flatten_and_ff: int =None, window=None):
        super().__init__()

        self.layer_1d = nn.ModuleList([
            ConvTranspose1D_ReLU(
                feature_dims[i],
                feature_dims[i+1],
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2
            ) for i in range(len(feature_dims)-1)
        ])

        self.layer_res = None
        if residual_dims is not None:
            self.layer_res = nn.ModuleList([
                ResidualBlock1D(
                    feature_dims[0], residual_dims[i]
                ) for i in range(len(residual_dims))
            ])

        self.output = []
        if flatten_and_ff is not None:
            self.ouptut = nn.ModuleList([
                nn.Linear(flatten_and_ff, feature_dims[0] * window),
                nn.Unflatten(1, (feature_dims[0], window))
            ])
    
    def forward(self, x):
        for l in self.output:
            x = l(x)
        for l in self.layer_res:
            x = l(x)
        for l in self.layer_1d:
            x = l(x)
        return x
    

class CollaborativeIDSNet(nn.Module):

    def __init__(self, network_encoder: nn.Module, host_encoder: nn.Module, embedding_encoder: nn.Module, aggregation_module: nn.Module):
        super().__init__()

        self.network_encoder = network_encoder
        self.host_encoder = nn.Sequential(host_encoder, nn.Flatten())
        self.embedding_encoder = nn.Sequential(embedding_encoder, nn.Flatten())
        self.aggregation_module = nn.Sequential(nn.ReLU(), aggregation_module)

    def forward(self, x_network: torch.Tensor, x_host: torch.Tensor, x_embeddings: torch.Tensor):

        x_network = self.network_encoder(x_network)
        x_host = self.host_encoder(x_host)
        x_embeddings = self.embedding_encoder(x_embeddings)


        x_agg = torch.cat([x_network, x_host, x_embeddings], dim=1)
        logits = self.aggregation_module(x_agg)

        return logits, x_network
    