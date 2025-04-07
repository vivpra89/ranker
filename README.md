# Deep & Cross Network v2 Re-Ranker

This project implements a re-ranker for recommender systems using the Deep & Cross Network v2 (DCN v2) architecture in PyTorch. The re-ranker takes multiple types of embeddings as input and learns to re-rank items based on their relevance.

## Quick Start

### Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

### Basic Usage

1. Prepare your input features in the following format:

   **User Features:**
   - User embeddings (als_user_embeds_binary): Binary embeddings from ALS
   - Demographics: age, gender, geo, region
   - User behavior: click, atc (add to cart), purchase, loose_atc, loose_purch
   - Age bins: age_bins, age_bins_double
   - Anonymous flag: anon_flag
   
   **Product Features:**
   - Product embeddings (als_prod_embeds_binary): Binary embeddings from ALS
   - Product metadata: product_gender, product_body_part
   - Taxonomy: product_taxonomy_id
   - Age descriptors: product_age_desc
   
   **Interaction Features:**
   - Style information: style_code, style_color
   - Anchor information: anchor_style, anchor_body_part, anchor_gender, anchor_taxonomy_id
   - Position features: position_in_carousel
   - Event data: event_date, event_timestamp, event_dow
   - Experience type and page details: experience_type, page_type, page_detail
   
   **System Features:**
   - Build information: app_build, app_version
   - System details: os_name, os_version
   - Raw UPM ID: raw_upm_id

2. Create a model instance:

```python
from src.models.dcn import DCNv2

model = DCNv2(config)  # See config.yml for full configuration options
```

3. Create datasets and data loaders:

```python
from torch.utils.data import DataLoader
from src.main import ReRankerDataset

train_dataset = ReRankerDataset(features, labels, config)
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
├───────────┬───────────┬────────────┬──────────┬───────────┬──────────┬────────┤
│  User     │ Product   │ Style &    │ Anchor   │  Event    │ Page &   │ System │
│ Features  │ Features  │ Position   │ Features │  Data     │  Exp.    │  Info  │
└─────┬─────┴────┬─────┴─────┬──────┴────┬─────┴─────┬─────┴────┬─────┴───┬────┘
      │          │           │            │           │          │         │
      ▼          ▼           ▼            ▼           ▼          ▼         ▼
┌──────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌───────┐ ┌────────┐ ┌─────┐
│ User     │ │ Product │ │ Style    │ │ Anchor  │ │ Event │ │ Page   │ │ Sys  │
│ Embedder │ │ Embedder│ │ Encoder  │ │ Encoder │ │ Proc. │ │ Proc.  │ │ Proc.│
└────┬─────┘ └────┬────┘ └────┬─────┘ └────┬────┘ └───┬───┘ └───┬────┘ └──┬──┘
     │           │           │           │           │         │         │
     └───────────┴───────────┴───────────┴───────────┴─────────┴─────────┘
                                        │
                                        ▼
                           ┌────────────────────────┐
                           │   Feature Fusion       │
                           │   & Normalization     │
                           └──────────┬────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │      Combined Features         │
                    └──────────────┬─────────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
         ┌──────────────────┐         ┌───────────────────┐
         │  Cross Network   │         │   Deep Network    │
         │                  │         │                   │
         │ Feature Crossing │         │ Non-linear       │
         │ (Low-rank DCN)   │         │ Patterns (MLP)   │
         └────────┬─────────┘         └────────┬────────┘
                 │                             │
                 └──────────────┬─────────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │  Feature Fusion   │
                     │  & Combination    │
                     └────────┬──────────┘
                              │
                              ▼
                     ┌───────────────────┐
                     │   Task Heads      │
                     └────────┬──────────┘
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
          ┌──────────┐  ┌──────────┐  ┌──────────┐
          │  Click   │  │ Purchase │  │ Add to   │
          │  Score   │  │  Score   │  │  Cart    │
          └──────────┘  └──────────┘  └──────────┘
```

The architecture processes these features through specialized components:

1. **Feature Processing Layer**:
   - User Embedder: Processes user embeddings and demographic data
   - Product Embedder: Handles product embeddings and metadata
   - Style Encoder: Processes style and position features
   - Anchor Encoder: Handles anchor-related features
   - Event Processor: Processes temporal and event data
   - Page Processor: Handles page and experience features
   - System Processor: Processes system and build information

2. **Feature Fusion**:
   - Combines all processed features
   - Applies normalization and feature alignment
   - Creates a unified representation

3. **Core Architecture**:
   - Cross Network: Models explicit feature interactions
   - Deep Network: Captures complex non-linear patterns

4. **Task Heads**:
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