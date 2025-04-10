import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import math

class ExpertLayer(nn.Module):
    """Single expert layer implementation."""
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TopKGating(nn.Module):
    """Top-k gating mechanism for routing inputs to experts."""
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        k: int,
        capacity_factor: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.dropout = nn.Dropout(dropout)
        self.wg = nn.Linear(input_dim, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of:
                - combine_weights: Shape [batch_size, k, num_experts]
                - dispatch_mask: Shape [batch_size, k, num_experts]
                - expert_mask: Shape [batch_size, num_experts]
        """
        batch_size = x.shape[0]
        
        # Calculate gates
        gates = self.dropout(x)
        gates = self.wg(gates)  # [batch_size, num_experts]
        gates = F.softmax(gates, dim=-1)
        
        # Calculate capacity
        capacity = int(self.capacity_factor * batch_size * self.k / self.num_experts)
        
        # Get top-k gates and indices
        top_k_gates, top_k_indices = torch.topk(gates, k=self.k, dim=-1)
        
        # Normalize top-k gates
        top_k_gates = F.softmax(top_k_gates, dim=-1)
        
        # Create dispatch mask
        dispatch_mask = torch.zeros_like(gates).scatter_(-1, top_k_indices, 1.0)
        
        # Apply capacity constraints
        expert_mask = torch.ones_like(gates)
        if self.training:
            position_in_expert = torch.cumsum(dispatch_mask, dim=0) * dispatch_mask
            expert_mask = (position_in_expert <= capacity).float()
            dispatch_mask = dispatch_mask * expert_mask
        
        # Create combine weights
        combine_weights = dispatch_mask * gates.unsqueeze(1)  # [batch_size, k, num_experts]
        
        return combine_weights, dispatch_mask, expert_mask

class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with top-k routing."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: Dict[str, Any]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = config['num_experts']
        self.k = config['k_experts']
        self.expert_hidden_size = config['expert_hidden_size']
        
        # Create experts
        self.experts = nn.ModuleList([
            ExpertLayer(
                input_dim=input_dim,
                hidden_size=config['expert_hidden_size'],
                output_dim=output_dim,
                dropout=config['expert_dropout']
            )
            for _ in range(self.num_experts)
        ])
        
        # Create gating network
        self.gating = TopKGating(
            input_dim=input_dim,
            num_experts=self.num_experts,
            k=self.k,
            capacity_factor=config['capacity_factor'],
            dropout=config['gating_dropout']
        )
        
        # Noise for load balancing
        self.noise_epsilon = 1e-2
        
    def _load_balance_loss(self, gates: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        """Calculate load balancing loss to encourage equal expert utilization."""
        # Calculate importance of each expert
        importance = gates.sum(0)
        
        # Add a small amount of noise to break symmetry
        noise = torch.randn_like(importance) * self.noise_epsilon
        importance = importance + noise
        
        # Calculate loss
        loss = torch.sum(importance * importance) * (self.num_experts / (torch.sum(importance) ** 2))
        
        return loss
        
    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            return_loss: Whether to return the load balancing loss
            
        Returns:
            Tuple of:
                - output: Shape [batch_size, output_dim]
                - loss: Optional load balancing loss
        """
        batch_size = x.shape[0]
        
        # Get routing weights and masks
        combine_weights, dispatch_mask, expert_mask = self.gating(x)
        
        # Dispatch to experts
        expert_inputs = x.unsqueeze(1).expand(-1, self.k, -1)  # [batch_size, k, input_dim]
        expert_outputs = torch.zeros(batch_size, self.k, self.output_dim, device=x.device)
        
        for i, expert in enumerate(self.experts):
            expert_mask_i = dispatch_mask[:, :, i].unsqueeze(-1)  # [batch_size, k, 1]
            if expert_mask_i.sum() > 0:
                expert_inputs_i = expert_inputs * expert_mask_i
                expert_outputs_i = expert(expert_inputs_i.reshape(-1, self.input_dim))
                expert_outputs += (
                    expert_outputs_i.reshape(batch_size, self.k, -1) * expert_mask_i
                )
        
        # Combine expert outputs
        output = torch.sum(expert_outputs * combine_weights.unsqueeze(-1), dim=1)
        
        # Calculate load balancing loss if requested
        loss = None
        if return_loss and self.training:
            loss = self._load_balance_loss(combine_weights.sum(1), expert_mask)
            
        return output, loss 