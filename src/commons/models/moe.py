import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_experts: int, dropout_rate: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_experts)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return softmax over expert dimension
        return F.softmax(self.network(x), dim=-1)

class MixtureOfExperts(nn.Module):
    def __init__(self, config: Dict[str, Any], input_dim: int, output_dim: int):
        super().__init__()
        self.config = config['model']['moe_config']
        self.num_experts = self.config['num_experts']
        self.use_region_specific = self.config['use_region_specific_experts']
        
        # Create global expert
        self.global_expert = ExpertNetwork(
            input_dim=input_dim,
            hidden_size=self.config['expert_hidden_size'],
            output_dim=output_dim,
            dropout_rate=self.config['expert_dropout']
        )
        
        # Create region-specific experts if enabled
        if self.use_region_specific:
            self.region_experts = nn.ModuleList([
                ExpertNetwork(
                    input_dim=input_dim,
                    hidden_size=self.config['expert_hidden_size'],
                    output_dim=output_dim,
                    dropout_rate=self.config['expert_dropout']
                ) for _ in range(self.num_experts - 1)  # -1 because we already have global expert
            ])
        
        # Create gating network
        self.gating_network = GatingNetwork(
            input_dim=input_dim,
            hidden_size=self.config['gating_hidden_size'],
            num_experts=self.num_experts,
            dropout_rate=self.config['expert_dropout']
        )
        
        self.training_stage = 'global'  # Can be 'global' or 'regional'
    
    def set_training_stage(self, stage: str):
        """Set the training stage ('global' or 'regional')."""
        assert stage in ['global', 'regional']
        self.training_stage = stage
        
        # During global training, only train global expert
        if stage == 'global':
            for param in self.global_expert.parameters():
                param.requires_grad = True
            if self.use_region_specific:
                for expert in self.region_experts:
                    for param in expert.parameters():
                        param.requires_grad = False
            for param in self.gating_network.parameters():
                param.requires_grad = False
        else:  # Regional training
            for param in self.global_expert.parameters():
                param.requires_grad = False
            if self.use_region_specific:
                for expert in self.region_experts:
                    for param in expert.parameters():
                        param.requires_grad = True
            for param in self.gating_network.parameters():
                param.requires_grad = True
    
    def forward(self, x: torch.Tensor, region_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = x.size(0)
        
        # Get expert outputs
        global_output = self.global_expert(x)
        expert_outputs = [global_output]
        
        if self.use_region_specific:
            regional_outputs = [expert(x) for expert in self.region_experts]
            expert_outputs.extend(regional_outputs)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_dim]
        
        # Get gating weights
        gating_weights = self.gating_network(x)  # [batch_size, num_experts]
        
        # Compute final output as weighted sum of expert outputs
        combined_output = torch.sum(
            expert_outputs * gating_weights.unsqueeze(-1),
            dim=1
        )
        
        # Compute load balancing loss
        # Encourage uniform expert utilization
        expert_usage = gating_weights.mean(0)
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        load_balancing_loss = F.kl_div(
            expert_usage.log(),
            target_usage,
            reduction='batchmean'
        )
        
        # During regional training, compute knowledge distillation loss
        kd_loss = torch.tensor(0.0, device=x.device)
        if self.training_stage == 'regional':
            with torch.no_grad():
                teacher_output = self.global_expert(x)
            kd_loss = F.mse_loss(combined_output, teacher_output)
        
        losses = {
            'load_balancing': load_balancing_loss * self.config['load_balancing_loss_weight'],
            'knowledge_distillation': kd_loss * self.config['knowledge_distillation_weight']
        }
        
        return combined_output, losses 