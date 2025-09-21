import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

from .nn import MLP

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Implementing the positional encoding to add onto the input of Transformer-based models
    """

    def __init__(self, d_model, dropout=0.1, max_len=100, batch_first=True):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        positional_encoding = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)

        if not batch_first:
            positional_encoding = positional_encoding.transpose(0, 1)

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :]

        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder as described in the original transformer paper with additional embedding and head layers to transform input or output
    """

    def __init__(self, n_layer, head=None, embedding=None, d_model=512, n_head=32, dim_feedforward=2048, dropout=0.1,
                  activation=F.relu, layer_norm_eps=1e-05, max_len=20, norm_first=True, bias=True, device=None, dtype=None):
        super(TransformerEncoder, self).__init__()

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, 
                                    batch_first=True, norm_first=norm_first, bias=bias, device=device, dtype=dtype) for _ in range(n_layer)])
        if embedding is None:
            embedding = nn.Identity()
        if isinstance(embedding, int):
            embedding = nn.Linear(embedding, d_model)
        self.embedding = embedding
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        if head is None:
            head = nn.Identity()
        if isinstance(head, int):
            head = nn.Linear(d_model, head)
        self.head = head

    def forward(self, src):
        # Expecting source to be of dimension BxNxL
        src = self.embedding(src)
        src = self.pos_encoding(src)
        for layer in self.transformer_layers:
            src = layer(src)
        src = self.head(src)
    
        return src

class AutoregressiveTransformerEncoder(nn.Module):
    """
    Transformer Encoder but with additional masking such that at each position no view into the feature by the model is possible
    """

    def __init__(self, n_layer, head=None, embedding=None, d_model=512, n_head=32, dim_feedforward=2048, dropout=0.1,
                  activation=F.relu, layer_norm_eps=1e-05, window_size=20, norm_first=True, bias=True, device=None, dtype=None):
        super(AutoregressiveTransformerEncoder, self).__init__()

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, 
                                    batch_first=True, norm_first=norm_first, bias=bias, device=device, dtype=dtype) for _ in range(n_layer)])
        if embedding is None:
            embedding = nn.Identity()
        self.embedding = embedding
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=window_size)

        if head is None:
            head = nn.Identity()
        self.head = head
    
    def forward(self, src, src_mask):
        # Expecting source to be of dimension BxNxL
        src = self.embedding(src)
        src = self.pos_encoding(src)
        for layer in self.transformer_layers:
            src = layer(src, src_mask=src_mask, is_causal=True)
        src = self.head(src)
    
        return src

        
class MaskedTransformerEncoder(nn.Module):
    """
    Encoder for BERT-style training. The input sequence is masked at different positions. To increase difficulty multiple consecutive positions
    are masked.
    """

    def __init__(self, n_layer, mean_masked_length: int = 3, masking_ratio: float = 0.15, embedding: nn.Module = None, head: nn.Module = None, d_model: int = 128, n_head: int = 8, dim_feedforward: int = 512, dropout: float = 0.1,
                 activation: callable = F.relu, layer_norm_eps: float = 1e-05, window_size: int = 20, norm_first: bool = True, bias: bool = True, device: str = None, dtype: torch.dtype = None):
        super(MaskedTransformerEncoder, self).__init__()

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps,
                                    batch_first=True, norm_first=norm_first, bias=bias, device=device, dtype=dtype) for _ in range(n_layer)])
        if embedding is None:
            embedding = nn.Identity()
        self.embedding = embedding
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=window_size)

        if head is None:
            head = nn.Identity()
        self.head = head

        self.mean_masked_length = mean_masked_length
        self.masking_ratio = masking_ratio

    def forward(self, src):

        mask = torch.ones_like(src)

        if self.training:

            batch_size, seq_len, _ = src.size()
            masked_indices = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=src.device)

            pos = torch.zeros(batch_size, dtype=torch.long, device=src.device)
            unmask_len = torch.distributions.Geometric(self.masking_ratio / ((1 - self.masking_ratio) * self.mean_masked_length + self.masking_ratio)).sample((batch_size,)).long().to(src.device)
            pos += unmask_len

            while torch.any(pos < seq_len):
                mask_len = torch.distributions.Geometric(1 / (self.mean_masked_length)).sample((batch_size,)).long().to(src.device) + 1
                mask_len = torch.min(mask_len, seq_len - pos)

                masked_indices_iteration = (pos.unsqueeze(1) <= torch.arange(seq_len, device=src.device).unsqueeze(0)) & (torch.arange(seq_len, device=src.device).unsqueeze(0) < (pos + mask_len).unsqueeze(1))        
                masked_indices = masked_indices | masked_indices_iteration

                pos += mask_len

                unmask_len = torch.distributions.Geometric(self.masking_ratio / ((1 - self.masking_ratio) * self.mean_masked_length)).sample((batch_size,)).long().to(src.device) + 1
                pos += unmask_len

            mask[masked_indices] = 0
        
        src = src * mask
        src = self.embedding(src)
        src = self.pos_encoding(src)
        for layer in self.transformer_layers:
            src = layer(src)
        
        src = self.head(src)
        return src, mask
        