# Deep & Cross Network v2 Re-Ranker

This project implements a re-ranker for recommender systems using the Deep & Cross Network v2 (DCN v2) architecture in PyTorch. The re-ranker takes multiple types of embeddings as input and learns to re-rank items based on their relevance.

## Quick Start

### Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

### Basic Usage

1. Prepare your embeddings data in the following format:
   - Product embeddings: shape (n_samples, product_embed_dim)
   - User embeddings: shape (n_samples, user_embed_dim)
   - Geo embeddings: shape (n_samples, geo_embed_dim)
   - Country embeddings: shape (n_samples, country_embed_dim)
   - Labels: shape (n_samples, 1) with binary values

2. Create a model instance:

```python
from src.models.dcn import DCNv2

model = DCNv2(
    product_embed_dim=128,
    user_embed_dim=64,
    geo_embed_dim=32,
    country_embed_dim=32,
    num_cross_layers=3,
    hidden_layers=[256, 128, 64]
)
```

3. Create datasets and data loaders:

```python
from torch.utils.data import DataLoader
from src.main import ReRankerDataset

train_dataset = ReRankerDataset(
    product_embeds,
    user_embeds,
    geo_embeds,
    country_embeds,
    labels
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

4. Train the model:

```python
from src.utils.trainer import ReRankerTrainer

trainer = ReRankerTrainer(model)
trainer.train(train_loader, val_loader, num_epochs=10)
```

## Architecture Overview

Below is a detailed architecture diagram of the DCN v2 Re-ranker:

```
                                     DCN v2 Re-Ranker Architecture
                                     ===========================

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Input Features                                      │
├───────────────┬───────────────┬────────────────┬──────────────┬───────────────┤
│   Product     │    User       │     Geo        │  Country     │  Interaction   │
│  Embeddings   │  Embeddings   │    Features    │  Features    │   History     │
└───────┬───────┴───────┬───────┴───────┬────────┴──────┬───────┴───────┬───────┘
        │               │               │                │               │
┌───────▼───────┐ ┌─────▼─────┐ ┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼─────┐
│   Feature     │ │  Feature   │ │    Text     │ │    Text     │ │ Sequence  │
│  Processing   │ │ Processing │ │  Embedder   │ │  Embedder   │ │  Encoder  │
└───────┬───────┘ └─────┬─────┘ └──────┬──────┘ └──────┬──────┘ └─────┬─────┘
        │               │               │                │               │
        └───────────────┴───────────────┴────────┬──────┴───────────────┘
                                                │
                                    ┌───────────▼────────────┐
                                    │    Feature Fusion      │
                                    └───────────┬────────────┘
                                                │
                              ┌─────────────────┴─────────────────┐
                              │                                   │
                     ┌────────▼──────────┐            ┌──────────▼─────────┐
                     │   Cross Network   │            │    Deep Network     │
                     │                   │            │                     │
                     │  Feature Crossing │            │ Non-linear Patterns │
                     │   (Low-rank DCN)  │            │   (MLP w/ Skip)    │
                     └────────┬──────────┘            └──────────┬─────────┘
                              │                                   │
                              └─────────────────┬─────────────────┘
                                               │
                                    ┌──────────▼───────────┐
                                    │    Feature Fusion    │
                                    └──────────┬───────────┘
                                               │
                                ┌──────────────┴──────────────┐
                                │        Task Heads           │
                        ┌───────┴──────┬────────┴───────┐
                        │              │                │
                   ┌────▼────┐    ┌────▼────┐     ┌────▼────┐
                   │  Click  │    │Purchase │     │Add to   │
                   │ Score   │    │ Score   │     │Cart     │
                   └─────────┘    └─────────┘     └─────────┘
```

The architecture consists of several key components:

1. **Input Features**: Multiple types of input features including product embeddings, user embeddings, geographical features, and interaction history.

2. **Feature Processing**: Each input type goes through specific processing:
   - Embeddings are processed through feature layers
   - Text features are processed through transformer-based embedders
   - Sequential data is handled by a dedicated sequence encoder

3. **Core Architecture**:
   - Cross Network: Efficiently models feature interactions using low-rank DCN
   - Deep Network: Captures complex patterns using deep MLP with skip connections

4. **Task Heads**: Multiple prediction heads for different tasks:
   - Click prediction
   - Purchase prediction
   - Add-to-cart prediction

## Technical Documentation

### 1. Architecture Overview

The re-ranker implements an enhanced version of the Deep & Cross Network v2 (DCN-v2) architecture, specifically designed for multi-task learning in recommendation systems. The architecture consists of three main components:

#### 1.1 Feature Processing Layer
- Handles multiple types of input features:
  - Pretrained embeddings (product, user)
  - Text features (geo, country) using transformer models
  - Sequential features (interaction history) using transformer encoders
  - Numeric features (price)
  - Categorical features (category)

#### 1.2 Core Architecture
- Parallel processing through two networks:
  1. Cross Network: Models explicit feature interactions
  2. Deep Network: Captures complex non-linear patterns
- Features are combined and fed through task-specific heads

#### 1.3 Output Layer
- Multiple task heads for different prediction tasks:
  - Click prediction
  - Purchase prediction
  - Add-to-cart prediction

### 2. Components in Detail

#### 2.1 Feature Processing Components

##### 2.1.1 TextEmbedder
```python
class TextEmbedder(nn.Module):
    def __init__(self, model_name: str, output_dim: int):
```
- Uses HuggingFace transformers for text processing
- Components:
  - Transformer encoder (e.g., DistilBERT)
  - Projection layers to desired dimension
  - BiasModule for learnable feature calibration
- Processing steps:
  1. Tokenization with truncation/padding
  2. Transformer encoding
  3. CLS token extraction
  4. Dimension projection

##### 2.1.2 CrossNetV2Layer
```python
class CrossNetV2Layer(nn.Module):
    def __init__(self, input_dim):
```
- Implements efficient feature crossing using low-rank decomposition
- Components:
  - Low-rank matrices U (input_dim × rank) and V (rank × input_dim)
  - Learnable bias term
  - Layer normalization
- Mathematical formulation:
  ```
  crossed = (xl × U) × V  # Low-rank approximation
  output = x0 * crossed + bias + xl  # Cross interaction with residual
  ```
- Memory efficiency: Reduces parameters from O(d²) to O(d*r)

##### 2.1.3 DeepNetwork
```python
class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers=[256, 128, 64], dropout_rate=0.1):
```
- Multi-layer feed-forward network with advanced features
- Components per layer:
  1. Linear transformation
  2. Layer normalization
  3. GELU activation
  4. Dropout
- Skip connection types:
  - Direct connections for matching dimensions
  - Learned projections for dimension mismatch
- Output projection with normalization

##### 2.1.4 TaskHead
```python
class TaskHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
```
- Task-specific prediction layers
- Components:
  - Multiple feed-forward layers
  - ReLU activation
  - Batch normalization
  - Dropout regularization
  - Sigmoid output activation

#### 2.2 Main Model (DCNv2)

```python
class DCNv2(nn.Module):
    def __init__(self, config: Dict[str, Any]):
```

##### Feature Processing
- Dynamic feature handling based on configuration
- Automatic dimension calculation
- Feature type-specific processing

##### Model Components
1. Text Embedders:
   - One per text feature
   - Configurable model selection
   - Dimension projection

2. Sequence Encoders:
   - Optional sequence processing
   - Transformer-based architecture
   - Configurable parameters

3. Cross Network:
   - Multiple cross layers
   - Stochastic depth regularization
   - Low-rank feature interactions

4. Deep Network:
   - Configurable layer dimensions
   - Skip connections
   - Advanced regularization

5. Task Heads:
   - One per prediction task
   - Task-specific architectures
   - Shared feature representation

### 3. Training Details

#### 3.1 Training Configuration
```yaml
training:
  optimizer:
    name: "adamw"
    learning_rate: 0.001
    weight_decay: 0.01
    layer_wise_lr: true
```

#### 3.2 Optimization Strategy

##### 3.2.1 Optimizer
- AdamW optimizer with weight decay
- Layer-wise learning rates
- Configurable parameters:
  - Base learning rate: 0.001
  - Weight decay: 0.01
  - Beta parameters: (0.9, 0.999)

##### 3.2.2 Learning Rate Schedule
```yaml
scheduler:
  enabled: true
  type: "cosine"
  warmup_steps: 1000
  min_lr: 1e-6
```
- Cosine annealing with warm-up
- Linear warm-up phase
- Minimum learning rate protection

#### 3.3 Training Features

##### 3.3.1 Regularization
- Multiple regularization techniques:
  - Dropout in deep network
  - Layer normalization
  - Stochastic depth in cross network
  - Weight decay
  - Gradient clipping

##### 3.3.2 Mixed Precision Training
```yaml
training_config:
  mixed_precision: true
```
- Automatic mixed precision (when available)
- Gradient scaling
- Memory efficiency

##### 3.3.3 Advanced Training Techniques
```yaml
training_config:
  gradient_centralization: true
  manifold_mixup: true
  mixup_alpha: 0.2
```
- Gradient centralization
- Manifold mixup regularization
- Early stopping mechanism

#### 3.4 Multi-Task Learning
```yaml
tasks:
  click:
    weight: 0.4
    loss: "bce"
  purchase:
    weight: 0.4
    loss: "bce"
  add_to_cart:
    weight: 0.2
    loss: "bce"
```
- Weighted task combination
- Task-specific metrics
- Binary cross-entropy loss
- Balanced task importance

#### 3.5 Monitoring and Logging
```yaml
logging:
  tensorboard:
    enabled: true
  checkpointing:
    save_best: true
    save_frequency: 5
```
- TensorBoard integration
- Model checkpointing
- Best model preservation
- Validation metrics tracking

## Project Structure

```
.
├── src/
│   ├── models/
│   │   └── dcn.py         # DCN v2 model implementation
│   ├── utils/
│   │   └── trainer.py     # Training utilities
│   └── main.py           # Example usage
├── requirements.txt
└── README.md
```

## License

MIT License 