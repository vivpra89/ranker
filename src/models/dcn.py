import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any, Union
from transformers import AutoModel, AutoTokenizer
from .sequence_encoder import SequenceEncoder
import numpy as np

class BiasModule(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))  # Weight vector
        self.bias = nn.Parameter(torch.zeros(size))    # Bias vector
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.weight + self.bias  # Add both weight and bias

class TextEmbedder(nn.Module):
    def __init__(self, model_name: str, output_dim: int):
        """
        Text to embedding converter using transformer models.
        
        Args:
            model_name: Pretrained model name from HuggingFace
            output_dim: Desired output embedding dimension
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Create projection as a ModuleList with a linear layer and a bias module
        self.projection = nn.ModuleList([
            nn.Linear(self.encoder.config.hidden_size, output_dim),  # First layer: full linear projection
            BiasModule(output_dim)  # Second layer: weight + bias term
        ])
    
    def forward(self, inputs: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        """Process text input and return embeddings."""
        if isinstance(inputs, torch.Tensor):
            # If input is already a tensor, convert to float
            embeddings = inputs.float()
        else:
            # Convert string input to list if needed
            if isinstance(inputs, str):
                inputs = [inputs]
            elif isinstance(inputs, (list, tuple)) and all(isinstance(x, dict) for x in inputs):
                # If inputs are dictionaries (product features), extract text values
                inputs = [x[next(iter(x))] for x in inputs]
            
            # Tokenize and get embeddings
            tokens = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            # Move tokens to the same device as the model
            tokens = {k: v.to(next(self.encoder.parameters()).device) for k, v in tokens.items()}
            outputs = self.encoder(**tokens)
            # Get CLS token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].float()
        
        # Apply projection layers
        x = self.projection[0](embeddings)  # Apply linear projection
        x = self.projection[1](x)  # Add weight and bias terms
        return x

class CrossNetV2Layer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        # Low-rank decomposition for memory efficiency
        self.rank = max(1, input_dim // 4)  # Reduce parameters while maintaining expressiveness
        self.U = nn.Parameter(torch.randn(input_dim, self.rank) / np.sqrt(self.rank))
        self.V = nn.Parameter(torch.randn(self.rank, input_dim) / np.sqrt(self.rank))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x0, xl):
        """
        Efficient implementation of feature crossing using low-rank matrix.
        
        Args:
            x0: Original input (batch_size, input_dim)
            xl: Output from previous layer (batch_size, input_dim)
        """
        # Compute cross interaction using low-rank decomposition
        # Instead of full matrix W (input_dim x input_dim), use U (input_dim x rank) and V (rank x input_dim)
        crossed = torch.matmul(torch.matmul(xl, self.U), self.V)  # More efficient than xl @ (U @ V)
        
        # Element-wise multiplication with original input (core DCN operation)
        cross_term = x0 * crossed
        
        # Add bias and residual connection
        output = cross_term + self.bias + xl
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        return output

class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers=[256, 128, 64], dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        
        layers = []
        skip_connections = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_layers):
            # Dense block
            block = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            layers.append(block)
            
            # Skip connection if dimensions match or need projection
            if i > 0:  # No skip connection for first layer
                if prev_dim == hidden_dim:
                    skip_connections.append((i-1, None))  # Direct connection
                else:
                    # Project previous layer output
                    skip_proj = nn.Sequential(
                        nn.Linear(hidden_layers[i-1], hidden_dim),
                        nn.LayerNorm(hidden_dim)
                    )
                    skip_connections.append((i-1, skip_proj))
            
            prev_dim = hidden_dim
        
        self.layers = nn.ModuleList(layers)
        self.skip_connections = skip_connections
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_layers[-1], hidden_layers[-1]),
            nn.LayerNorm(hidden_layers[-1]),
            nn.GELU()
        )
        
    def forward(self, x):
        layer_outputs = []
        out = x
        
        for i, layer in enumerate(self.layers):
            # Apply dense block
            layer_out = layer(out)
            
            # Add skip connection if available
            if i > 0:
                skip_info = self.skip_connections[i-1]
                prev_idx, skip_proj = skip_info
                
                if skip_proj is None:
                    # Direct connection
                    layer_out = layer_out + layer_outputs[prev_idx]
                else:
                    # Project previous layer output
                    layer_out = layer_out + skip_proj(layer_outputs[prev_idx])
            
            layer_outputs.append(layer_out)
            out = layer_out
        
        return self.output_proj(out)

class TaskHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.head = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.head(x)

class DCNv2(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary containing model and feature settings
        """
        super().__init__()
        self.feature_config = config['features']['feature_configs']
        self.enabled_features = config['features']['enabled_features']
        self.tasks = [task for task, spec in config['tasks'].items() if spec['enabled']]
        
        dcn_config = config['model']['dcn_config']
        
        # Initialize text embedders for text features
        self.text_embedders = nn.ModuleDict()
        
        # Initialize sequence encoders for sequence features
        self.sequence_encoders = nn.ModuleDict()
        
        for name in self.enabled_features:
            feature_config = self.feature_config[name]
            if not feature_config['enabled']:
                continue
                
            if feature_config['type'] == 'text':
                self.text_embedders[name] = TextEmbedder(
                    feature_config.get('model_name', 'distilbert-base-uncased'),
                    feature_config['dim']
                )
            elif feature_config['type'] == 'sequence':
                self.sequence_encoders[name] = SequenceEncoder(
                    input_dim=feature_config['input_dim'],
                    output_dim=feature_config['dim'],
                    nhead=feature_config.get('nhead', 4),
                    num_layers=feature_config.get('num_layers', 2),
                    dim_feedforward=feature_config.get('dim_feedforward', 256),
                    dropout=feature_config.get('dropout', dcn_config['dropout_rate']),
                    max_seq_length=feature_config.get('max_seq_length', 50),
                    activation=dcn_config['activation']
                )
        
        # Calculate total input dimension from enabled features
        self.total_embed_dim = sum(
            config['dim'] 
            for name, config in self.feature_config.items()
            if name in self.enabled_features and config['enabled']
        )
        
        # Cross Network
        self.cross_layers = nn.ModuleList([
            CrossNetV2Layer(self.total_embed_dim) 
            for _ in range(dcn_config['cross_layers'])
        ])
        
        # Deep Network
        self.deep_network = DeepNetwork(
            self.total_embed_dim, 
            dcn_config['hidden_layers'],
            dcn_config['dropout_rate']
        )
        
        # Task-specific heads
        combined_dim = self.total_embed_dim + dcn_config['hidden_layers'][-1]
        self.task_heads = nn.ModuleDict({
            task: TaskHead(combined_dim, dcn_config['task_hidden_layers'])
            for task in self.tasks
        })
        
        # Stochastic depth for cross layers
        self.stochastic_depth_rate = dcn_config['stochastic_depth_rate']
        self.training = True
    
    def process_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process input features based on their type."""
        processed = {}
        
        for name in self.enabled_features:
            if name not in features or not self.feature_config[name]['enabled']:
                continue
                
            feature_type = self.feature_config[name]['type']
            if feature_type == 'pretrained_embedding':
                # Use pretrained embeddings as is
                processed[name] = features[name]
            elif feature_type == 'text':
                # Convert text to embeddings
                text_input = features[name]
                if isinstance(text_input, str):
                    text_input = [text_input]
                elif isinstance(text_input, torch.Tensor):
                    # Already processed by inference code
                    processed[name] = text_input
                    continue
                
                # Tokenize text
                tokens = self.text_embedders[name].tokenizer(
                    text_input,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                # Move tokens to the same device as the model
                tokens = {k: v.to(next(self.text_embedders[name].encoder.parameters()).device) for k, v in tokens.items()}
                outputs = self.text_embedders[name].encoder(**tokens)
                # Get CLS token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :].float()
                
                # Apply projection layers
                x = self.text_embedders[name].projection[0](embeddings)  # Apply linear projection
                x = self.text_embedders[name].projection[1](x)  # Add weight and bias terms
                processed[name] = x
            elif feature_type == 'sequence':
                # Process sequence data
                sequence_data = features[name]
                if isinstance(sequence_data, tuple):
                    # If sequence data includes padding mask
                    sequences, padding_mask = sequence_data
                else:
                    sequences = sequence_data
                    padding_mask = None
                
                processed[name] = self.sequence_encoders[name](
                    sequences, 
                    padding_mask=padding_mask
                )
            else:
                # Numeric or categorical features
                processed[name] = features[name]
        
        return processed
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            features: Dictionary of input features
            
        Returns:
            Dictionary of task predictions
        """
        # Process features
        processed_features = self.process_features(features)
        
        # Concatenate all features
        x = torch.cat([processed_features[name] for name in self.enabled_features 
                      if name in processed_features], dim=1)
        
        # Cross network with stochastic depth
        x0 = x
        xl = x
        for layer in self.cross_layers:
            if self.training and torch.rand(1) < self.stochastic_depth_rate:
                continue
            xl = layer(x0, xl)
        
        # Deep network
        deep_out = self.deep_network(x)
        
        # Combine cross and deep outputs
        combined = torch.cat([xl, deep_out], dim=1)
        
        # Task-specific predictions
        return {
            task: self.task_heads[task](combined)
            for task in self.tasks
        }

    def predict_scores(
        self,
        user_features: Dict[str, Any],
        product_features: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """
        Predict scores for a list of products based on user features.
        
        Args:
            user_features: Dictionary of user features
            product_features: List of product feature dictionaries
            
        Returns:
            Dictionary of predicted scores for each task
        """
        # Convert features to tensors and move to device
        batch_features = {}
        
        # Process user features
        for name, value in user_features.items():
            if name not in self.enabled_features or not self.feature_config[name]['enabled']:
                continue
                
            feature_type = self.feature_config[name]['type']
            if feature_type == 'text':
                # Text feature - use embedder
                value = self.text_embedders[name]([value])
            elif isinstance(value, dict) and value.get('__type__') == 'ndarray':
                # Numeric feature from numpy array format
                value = torch.tensor(value['data']).float().reshape(value['shape'])
            elif isinstance(value, np.ndarray):
                # Numeric feature from numpy array
                value = torch.tensor(value).float()
            else:
                continue
                
            # Repeat user features for each product
            value = value.unsqueeze(0).repeat(len(product_features), 1)
            batch_features[name] = value.to(self.device)
        
        # Process product features
        for name in product_features[0].keys():
            if name not in self.enabled_features or not self.feature_config[name]['enabled']:
                continue
                
            feature_type = self.feature_config[name]['type']
            if feature_type == 'text':
                # Text feature - use embedder
                values = [p[name] for p in product_features]
                value = self.text_embedders[name](values)
            elif isinstance(product_features[0][name], dict) and product_features[0][name].get('__type__') == 'ndarray':
                # Numeric feature from numpy array format
                values = [p[name]['data'] for p in product_features]
                shapes = [p[name]['shape'] for p in product_features]
                value = torch.tensor(np.stack(values)).float()
                value = value.reshape(len(product_features), *shapes[0])
            elif isinstance(product_features[0][name], np.ndarray):
                # Numeric feature from numpy array
                values = [p[name] for p in product_features]
                value = torch.tensor(np.stack(values)).float()
            else:
                continue
                
            batch_features[name] = value.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.forward(batch_features)
            
        # Convert to numpy
        return {
            task: outputs[task].cpu().numpy()
            for task in outputs
        } 