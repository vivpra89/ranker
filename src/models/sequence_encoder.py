import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_length: int = 50,
        activation: str = "gelu"
    ):
        """
        Transformer-based sequence encoder for product interaction sequences.
        
        Args:
            input_dim: Dimension of input embeddings
            output_dim: Dimension of output embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            activation: Activation function to use
        """
        super().__init__()
        
        # Make sure input_dim is divisible by nhead
        self.d_model = (input_dim // nhead) * nhead
        if self.d_model != input_dim:
            self.input_proj = nn.Linear(input_dim, self.d_model)
        else:
            self.input_proj = nn.Identity()
            
        self.pos_encoder = PositionalEncoding(
            self.d_model, 
            max_seq_length, 
            dropout
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Project to output dimension
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Mask token for padding
        self.register_buffer(
            'mask_token',
            torch.zeros(1, 1, self.d_model)
        )
        
    def forward(
        self,
        sequences: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            sequences: Tensor of shape [batch_size, seq_len, input_dim]
            padding_mask: Boolean mask for padded positions (True for pad tokens)
                        shape: [batch_size, seq_len]
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        # Project input if necessary
        x = self.input_proj(sequences)
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Use mean pooling over sequence length
        if padding_mask is not None:
            # Create mask for averaging (False -> 1.0, True -> 0.0)
            avg_mask = (~padding_mask).float().unsqueeze(-1)
            # Mean pooling (considering only non-padded elements)
            x = (x * avg_mask).sum(dim=1) / avg_mask.sum(dim=1)
        else:
            # Simple mean pooling if no padding mask
            x = x.mean(dim=1)
        
        # Project to output dimension
        return self.output_proj(x)
        
    def encode_sequence(
        self,
        product_embeddings: torch.Tensor,
        sequence_indices: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a sequence of product indices using their embeddings.
        
        Args:
            product_embeddings: Tensor of shape [num_products, embedding_dim]
            sequence_indices: Tensor of shape [batch_size, seq_len]
            padding_mask: Optional boolean mask for padded positions
        
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        # Gather embeddings for the sequence
        sequence_embeddings = product_embeddings[sequence_indices]
        
        # Forward pass through transformer
        return self.forward(sequence_embeddings, padding_mask) 