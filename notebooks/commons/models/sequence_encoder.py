# Databricks notebook source
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

class SequenceEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 max_seq_length: int = 50,
                 transformer_config: Optional[Dict[str, Any]] = None):
        """
        Sequence encoder using transformer architecture.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            max_seq_length: Maximum sequence length
            transformer_config: Configuration for transformer layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Default transformer config
        if transformer_config is None:
            transformer_config = {
                'num_layers': 2,
                'num_heads': 4,
                'dropout': 0.1,
                'activation': 'gelu'
            }
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_length, hidden_dim))
        nn.init.xavier_uniform_(self.pos_encoding)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_config['num_heads'],
            dim_feedforward=4 * hidden_dim,
            dropout=transformer_config['dropout'],
            activation=transformer_config['activation']
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_config['num_layers']
        )
        
        # Output pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            padding_mask: Optional padding mask of shape (batch_size, seq_length)
                        where True indicates padding positions
        
        Returns:
            Encoded sequence of shape (batch_size, hidden_dim)
        """
        # Project input
        x = self.input_projection(x)  # (batch_size, seq_length, hidden_dim)
        
        # Add positional encoding
        seq_length = x.size(1)
        x = x + self.pos_encoding[:, :seq_length, :]
        
        # Transpose for transformer
        x = x.transpose(0, 1)  # (seq_length, batch_size, hidden_dim)
        
        # Apply transformer
        if padding_mask is not None:
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)
        
        # Transpose back
        x = x.transpose(0, 1)  # (batch_size, seq_length, hidden_dim)
        
        # Apply attention pooling
        attention_weights = self.attention_pooling(x)  # (batch_size, seq_length, 1)
        if padding_mask is not None:
            attention_weights = attention_weights.masked_fill(padding_mask.unsqueeze(-1), 0)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
        
        # Pool sequence
        x = (x * attention_weights).sum(dim=1)  # (batch_size, hidden_dim)
        
        return x 