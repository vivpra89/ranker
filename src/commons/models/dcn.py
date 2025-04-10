import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any, Union
from transformers import AutoModel, AutoTokenizer
from .sequence_encoder import SequenceEncoder
import numpy as np
from .moe import MixtureOfExperts

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
    def __init__(self, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = None
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = None
        self.register_parameter('U', None)
        self.register_parameter('V', None)

    def initialize_parameters(self, input_dim):
        self.input_dim = input_dim
        self.head_dim = input_dim // self.num_heads
        
        # Initialize parameters for each head
        U = torch.Tensor(input_dim, input_dim)  # Changed dimensions to match input
        V = torch.Tensor(input_dim, input_dim)  # Changed dimensions to match input
        
        # Xavier initialization
        nn.init.xavier_uniform_(U)
        nn.init.xavier_uniform_(V)
        
        # Register parameters properly
        self.register_parameter('U', nn.Parameter(U))
        self.register_parameter('V', nn.Parameter(V))

    def forward(self, x0, xl):
        if self.U is None:
            self.initialize_parameters(x0.size(1))
        elif self.input_dim != x0.size(1):
            self.initialize_parameters(x0.size(1))
        
        # Multi-head cross attention
        xl = F.dropout(xl, p=self.dropout, training=self.training)
        
        # Compute cross interactions more efficiently
        crossed = torch.matmul(torch.matmul(xl, self.U), self.V)
        out = x0 * crossed + xl
        
        return out

def get_activation(name):
    """Get activation function by name."""
    if name.lower() == 'relu':
        return nn.ReLU()
    elif name.lower() == 'gelu':
        return nn.GELU()
    elif name.lower() == 'selu':
        return nn.SELU()
    elif name.lower() == 'elu':
        return nn.ELU()
    elif name.lower() == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")

class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.1, activation='relu', use_batch_norm=True, use_layer_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        
        # Activation function
        self.activation = get_activation(activation)
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class TaskNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.1, activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class DCNv2(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary containing model and feature settings
        """
        super().__init__()
        self.config = config
        self.feature_config = config['features']['feature_configs']
        self.enabled_features = config['features']['enabled_features']
        self.tasks = [task for task, task_config in config['tasks'].items() if task_config['enabled']]
        
        # Initialize embedders and encoders
        self.categorical_embedders = nn.ModuleDict()
        self.text_embedders = nn.ModuleDict()
        self.sequence_encoders = nn.ModuleDict()
        
        # Initialize embedders and calculate dimensions
        self._initialize_embedders()
        
        # Initialize networks (will be done in first forward pass)
        self.cross_net = None
        self.deep_network = None
        self.task_networks = None
        
        # Stochastic depth rate
        dcn_config = config['model']['dcn_config']
        self.stochastic_depth_rate = dcn_config.get('stochastic_depth_rate', 0.0)
        
        # Set device
        self.device = torch.device(config.get('device', 'cpu'))
        self.to(self.device)
        
        # Initialize MoE if enabled
        self.use_moe = (
            config['model']['architecture'] == 'dcn_v2_moe' and 
            config['model']['moe_config']['enabled']
        )
        if self.use_moe:
            self.moe = MixtureOfExperts(
                config=config,
                input_dim=self.config['model']['dcn_config']['hidden_layers'][-1],
                output_dim=self.config['model']['dcn_config']['hidden_layers'][-1]
            )
    
    def _initialize_embedders(self):
        """Initialize embedders for different feature types."""
        for name in self.enabled_features:
            if not self.feature_config[name]['enabled']:
                continue
                
            feature_config = self.feature_config[name]
            feature_type = feature_config['type']
            feature_dim = feature_config['dim']
            
            if feature_type == 'categorical':
                num_categories = feature_config['num_categories']
                self.categorical_embedders[name] = nn.Embedding(num_categories, feature_dim)
            elif feature_type == 'text':
                self.text_embedders[name] = TextEmbedder(
                    model_name=feature_config.get('model_name', 'bert-base-uncased'),
                    output_dim=feature_dim
                )
            elif feature_type == 'sequence':
                if not feature_config.get('enabled', True):
                    continue
                self.sequence_encoders[name] = SequenceEncoder(
                    input_dim=feature_config['input_dim'],
                    hidden_dim=feature_config['dim'],
                    max_seq_length=feature_config.get('max_seq_length', 50),
                    transformer_config=feature_config.get('transformer_config', {})
                )
    
    def _initialize_networks(self, input_dim: int):
        """Initialize cross and deep networks."""
        # Cross network
        self.cross_net = nn.ModuleList([
            CrossNetV2Layer(
                num_heads=self.config['model']['dcn_config']['num_heads'],
                dropout=self.config['model']['dcn_config']['cross_dropout']
            )
            for _ in range(self.config['model']['dcn_config']['cross_layers'])
        ])
        
        # Deep network
        self.deep_network = DeepNetwork(
            input_dim=input_dim,
            hidden_layers=self.config['model']['dcn_config']['hidden_layers'],
            dropout_rate=self.config['model']['dcn_config']['dropout_rate'],
            activation=self.config['model']['dcn_config']['activation'],
            use_batch_norm=self.config['model']['dcn_config']['use_batch_norm'],
            use_layer_norm=self.config['model']['dcn_config']['use_layer_norm']
        )
        
        # Combined dimension after concatenating cross and deep outputs
        combined_dim = input_dim + self.config['model']['dcn_config']['hidden_layers'][-1]
        
        # Initialize MoE if enabled
        if self.use_moe:
            self.moe = MixtureOfExperts(
                config=self.config,
                input_dim=combined_dim,
                output_dim=combined_dim
            )
        
        # Task-specific networks
        self.task_networks = nn.ModuleDict()
        for task in self.tasks:
            if self.config['tasks'][task]['enabled']:
                self.task_networks[task] = TaskNetwork(
                    input_dim=combined_dim,
                    hidden_layers=self.config['model']['dcn_config']['task_hidden_layers'],
                    dropout_rate=self.config['model']['dcn_config']['dropout_rate'],
                    activation=self.config['model']['dcn_config']['activation']
                )
        
        # Move to correct device
        self.cross_net = self.cross_net.to(self.device)
        self.deep_network = self.deep_network.to(self.device)
        self.task_networks = self.task_networks.to(self.device)

    def process_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process input features based on their type."""
        processed = {}
        
        for name in self.enabled_features:
            if name not in features or not self.feature_config[name]['enabled']:
                continue
                
            feature_type = self.feature_config[name]['type']
            feature_dim = self.feature_config[name]['dim']
            
            if feature_type == 'pretrained_embedding':
                # Use pretrained embeddings as is
                processed[name] = features[name]
            elif feature_type == 'binary_embedding':
                # Use binary embeddings as is
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
                if not self.feature_config[name].get('enabled', True):
                    continue
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
            elif feature_type == 'categorical':
                # Convert categorical features to embeddings
                processed[name] = self.categorical_embedders[name](features[name])
            else:
                # Numeric features
                processed[name] = features[name].unsqueeze(-1) if features[name].dim() == 1 else features[name]
        
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
        feature_list = []
        for name in self.enabled_features:
            if name in processed_features:
                feature = processed_features[name]
                feature_list.append(feature)
        
        # Concatenate all features
        x = torch.cat(feature_list, dim=1)
        input_dim = x.size(1)
        
        # Initialize networks if not done yet or if input dimension changed
        if self.cross_net is None or self.deep_network is None or self.deep_network.input_dim != input_dim:
            self._initialize_networks(input_dim)
        
        # Cross network with stochastic depth
        x0 = x
        xl = x
        for i, layer in enumerate(self.cross_net):
            if self.training and torch.rand(1) < self.stochastic_depth_rate:
                continue
            xl = layer(x0, xl)
        
        # Deep network
        deep_out = self.deep_network(x)
        
        # Combine cross and deep outputs
        combined = torch.cat([xl, deep_out], dim=1)
        
        # Apply MoE if enabled
        moe_losses = {}
        if self.use_moe:
            region_ids = features.get('region', None)
            combined, moe_losses = self.moe(combined, region_ids)
        
        # Get task predictions
        predictions = {}
        for task in self.tasks:
            predictions[task] = self.task_networks[task](combined)
        
        # Add MoE losses if present
        if moe_losses:
            predictions.update(moe_losses)
        
        return predictions

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

    def set_moe_training_stage(self, stage: str):
        """Set the training stage for MoE if enabled."""
        if self.use_moe:
            self.moe.set_training_stage(stage) 