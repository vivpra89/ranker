import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader

from ..training.evaluate import (
    compute_ndcg,
    compute_map,
    compute_precision_at_k,
    compute_recall_at_k
)

def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int = None) -> float:
    """Compute Normalized Discounted Cumulative Gain."""
    if k is None:
        k = y_true.shape[0]
    
    # Get sorting indices in descending order of predictions
    pred_indices = np.argsort(y_pred)[::-1]
    
    # Calculate DCG
    dcg = np.sum([
        (2 ** y_true[idx] - 1) / np.log2(rank + 2)
        for rank, idx in enumerate(pred_indices[:k])
    ])
    
    # Calculate IDCG
    ideal_indices = np.argsort(y_true)[::-1]
    idcg = np.sum([
        (2 ** y_true[idx] - 1) / np.log2(rank + 2)
        for rank, idx in enumerate(ideal_indices[:k])
    ])
    
    return dcg / idcg if idcg > 0 else 0.0

def compute_mrr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank."""
    # Get sorting indices in descending order of predictions
    pred_indices = np.argsort(y_pred)[::-1]
    
    # Find the rank of the first relevant item
    for rank, idx in enumerate(pred_indices):
        if y_true[idx] == 1:
            return 1.0 / (rank + 1)
    return 0.0

def compute_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute Precision@k for binary relevance."""
    # Get top k indices
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    # Count relevant items in top k
    return np.mean(y_true[top_k_indices])

def compute_average_precision(y_true: np.ndarray, y_pred: np.ndarray, k: int = None) -> float:
    """Compute Average Precision (AP) for binary relevance."""
    if k is None:
        k = len(y_true)
    
    # Get sorting indices in descending order
    pred_indices = np.argsort(y_pred)[::-1][:k]
    
    # Calculate precision at each position where a relevant item was found
    precisions = []
    num_relevant = 0
    
    for i, idx in enumerate(pred_indices):
        if y_true[idx] == 1:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    
    if not precisions:  # No relevant items found
        return 0.0
    
    return np.mean(precisions)

def compute_map_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute Mean Average Precision at k (MAP@k)."""
    if len(y_true.shape) == 1:
        return compute_average_precision(y_true, y_pred, k)
    
    # For multiple queries
    aps = []
    for i in range(len(y_true)):
        ap = compute_average_precision(y_true[i], y_pred[i], k)
        aps.append(ap)
    return np.mean(aps)

def compute_recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute Recall@k for binary relevance."""
    # Get top k indices
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    # Count relevant items in top k
    num_relevant_in_top_k = np.sum(y_true[top_k_indices])
    # Total number of relevant items
    total_relevant = np.sum(y_true)
    
    if total_relevant == 0:
        return 0.0
    
    return num_relevant_in_top_k / total_relevant

class ReRankerTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device=None
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get enabled tasks and their weights
        self.tasks = [task for task, spec in config['tasks'].items() if spec['enabled']]
        self.task_weights = {
            task: spec['weight']
            for task, spec in config['tasks'].items()
            if spec['enabled']
        }
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config['training']['training_config']['mixed_precision'] else None
        
        # Setup logging
        self.setup_logging()
        
        # Initialize early stopping
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        self.patience_counter = 0
        self.best_epochs = []
        self.min_delta = float(config['training']['training_config'].get('early_stopping_min_delta', 1e-4))
        self.patience = int(config['training']['training_config'].get('early_stopping_patience', 3))
        
        # Initialize loss function for each task
        self.loss_fns = {}
        for task in self.tasks:
            loss_type = config['tasks'][task]['loss'].lower()
            if loss_type == 'bce':
                self.loss_fns[task] = nn.BCELoss()  # Using BCELoss since we apply sigmoid in model
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # MoE specific settings
        self.use_moe = (
            config['model']['architecture'] == 'dcn_v2_moe' and 
            config['model']['moe_config']['enabled']
        )
        if self.use_moe:
            self.moe_config = config['model']['moe_config']
            self.global_epochs = self.moe_config['training_stages']['global_expert_epochs']
            self.regional_epochs = self.moe_config['training_stages']['regional_expert_epochs']
    
    def setup_logging(self):
        """Setup logging and monitoring tools."""
        if self.config['logging']['tensorboard']['enabled']:
            log_dir = Path(self.config['logging']['tensorboard']['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
            
        if self.config['logging']['wandb']['enabled']:
            wandb.init(
                project=self.config['logging']['wandb']['project'],
                entity=self.config['logging']['wandb']['entity'],
                config=self.config
            )
        
        # Setup checkpointing directory
        if self.config['logging']['checkpointing']['save_best']:
            checkpoint_dir = Path(self.config['paths']['model_save_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = checkpoint_dir
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute metrics for each task."""
        metrics = {}
        
        for task in self.tasks:
            if task not in outputs or task not in labels:
                continue
                
            y_pred = outputs[task].cpu().numpy()
            y_true = labels[task].cpu().numpy()
            
            # Compute metrics based on task configuration
            task_metrics = self.config['tasks'][task]['metrics']
            
            for metric in task_metrics:
                metric_name = f'{task}_{metric}'
                if metric == 'auc':
                    metrics[metric_name] = roc_auc_score(y_true, y_pred)
                elif metric == 'accuracy':
                    metrics[metric_name] = accuracy_score(y_true, y_pred > 0.5)
                elif metric == 'precision':
                    metrics[metric_name] = precision_score(y_true, y_pred > 0.5)
                elif metric == 'recall':
                    metrics[metric_name] = recall_score(y_true, y_pred > 0.5)
                elif metric == 'ndcg':
                    # Compute NDCG at different K values
                    for k in [5, 10, 20]:
                        metrics[f'{metric_name}@{k}'] = compute_ndcg(y_true, y_pred, k=k)
                elif metric == 'mrr':
                    metrics[metric_name] = compute_mrr(y_true, y_pred)
                elif metric == 'map':
                    # Compute MAP at different K values
                    for k in [5, 10, 20]:
                        metrics[f'{metric_name}@{k}'] = compute_map_at_k(y_true, y_pred, k=k)
                elif metric == 'precision_at_k':
                    # Compute Precision at different K values
                    for k in [1, 3, 5, 10]:
                        metrics[f'{task}_p@{k}'] = compute_precision_at_k(y_true, y_pred, k=k)
                elif metric == 'recall_at_k':
                    # Compute Recall at different K values
                    for k in [5, 10, 20]:
                        metrics[f'{task}_r@{k}'] = compute_recall_at_k(y_true, y_pred, k=k)
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """Log metrics to all enabled logging tools."""
        # Log to tensorboard
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f"{prefix}{name}", value, step)
        
        # Log to wandb
        if self.config['logging']['wandb']['enabled']:
            wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint with comprehensive metrics."""
        if not self.config['logging']['checkpointing']['save_best'] and not is_best:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'config': self.config,  # Save config for reproducibility
            'tasks': self.tasks,
            'task_weights': self.task_weights
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_model_path = Path(self.config['paths']['best_model_path'])
            torch.save(checkpoint, best_model_path)
            
            # Also save a backup of the best model
            best_backup_path = best_model_path.parent / f"best_model_backup_epoch_{epoch}.pt"
            torch.save(checkpoint, best_backup_path)
        
        # Remove old checkpoints if needed
        max_checkpoints = self.config['logging']['checkpointing']['max_checkpoints']
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if len(checkpoints) > max_checkpoints:
            for checkpoint in checkpoints[:-max_checkpoints]:
                checkpoint.unlink()
    
    def should_stop_early(self, val_metrics: Dict[str, float], epoch: int) -> bool:
        """Check if training should stop early."""
        if self.patience is None:
            return False
        
        # Get validation loss
        current_val_loss = float(val_metrics['total_loss'])
        
        # Update best validation loss
        if self.best_val_loss is None or current_val_loss < float(self.best_val_loss):
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Check if we should stop
        if self.patience_counter >= self.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            return True
        
        return False
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        task_losses = {task: 0.0 for task in self.tasks}
        all_outputs = {task: [] for task in self.tasks}
        all_labels = {task: [] for task in self.tasks}
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, (features, labels) in enumerate(pbar):
                # Move features to device
                features = {
                    name: value.to(self.device) if isinstance(value, torch.Tensor)
                    else (tuple(v.to(self.device) for v in value) if isinstance(value, tuple)
                          else value)
                    for name, value in features.items()
                }
                labels = {name: label.to(self.device) for name, label in labels.items()}
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision if enabled
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(features)
                        loss = 0
                        for task in self.tasks:
                            task_loss = self.loss_fns[task](outputs[task], labels[task])
                            loss += self.task_weights[task] * task_loss
                            task_losses[task] += task_loss.item()
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping if enabled
                    if self.config['training']['training_config']['gradient_clipping'] > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training']['training_config']['gradient_clipping']
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard forward pass
                    outputs = self.model(features)
                    loss = 0
                    for task in self.tasks:
                        task_loss = self.loss_fns[task](outputs[task], labels[task])
                        loss += self.task_weights[task] * task_loss
                        task_losses[task] += task_loss.item()
                    
                    # Standard backward pass
                    loss.backward()
                    
                    # Gradient clipping if enabled
                    if self.config['training']['training_config']['gradient_clipping'] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training']['training_config']['gradient_clipping']
                        )
                    
                    self.optimizer.step()
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Collect outputs and labels for metrics
                for task in self.tasks:
                    if task in outputs and task in labels:
                        all_outputs[task].append(outputs[task].detach())
                        all_labels[task].append(labels[task])
                
                # Update total loss
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Compute epoch metrics
        epoch_metrics = {}
        for task in self.tasks:
            if len(all_outputs[task]) > 0 and len(all_labels[task]) > 0:
                task_outputs = torch.cat(all_outputs[task])
                task_labels = torch.cat(all_labels[task])
                task_metrics = self.compute_metrics(
                    {task: task_outputs},
                    {task: task_labels}
                )
                epoch_metrics.update(task_metrics)
                epoch_metrics[f'{task}_loss'] = task_losses[task] / num_batches
        
        epoch_metrics['total_loss'] = total_loss / num_batches
        return epoch_metrics
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        task_losses = {task: 0.0 for task in self.tasks}
        all_outputs = {task: [] for task in self.tasks}
        all_labels = {task: [] for task in self.tasks}
        num_batches = len(val_loader)
        
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation") as pbar:
                for batch_idx, (features, labels) in enumerate(pbar):
                    # Move features to device
                    features = {
                        name: value.to(self.device) if isinstance(value, torch.Tensor)
                        else (tuple(v.to(self.device) for v in value) if isinstance(value, tuple)
                              else value)
                        for name, value in features.items()
                    }
                    labels = {name: label.to(self.device) for name, label in labels.items()}
                    
                    # Forward pass
                    outputs = self.model(features)
                    loss = 0
                    for task in self.tasks:
                        task_loss = self.loss_fns[task](outputs[task], labels[task])
                        loss += self.task_weights[task] * task_loss
                        task_losses[task] += task_loss.item()
                        
                        all_outputs[task].append(outputs[task])
                        all_labels[task].append(labels[task])
                    
                    # Update total loss
                    total_loss += loss.item()
                    avg_loss = total_loss / (batch_idx + 1)
                    if pbar:
                        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Compute validation metrics
        val_metrics = {}
        for task in self.tasks:
            if len(all_outputs[task]) > 0 and len(all_labels[task]) > 0:
                task_outputs = torch.cat(all_outputs[task])
                task_labels = torch.cat(all_labels[task])
                task_metrics = self.compute_metrics(
                    {task: task_outputs},
                    {task: task_labels}
                )
                val_metrics.update(task_metrics)
                val_metrics[f'{task}_loss'] = task_losses[task] / num_batches
        
        val_metrics['total_loss'] = total_loss / num_batches
        return val_metrics
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> float:
        """Train the model."""
        best_val_metric = float('inf')
        patience_counter = 0
        
        # Get number of epochs and patience from config
        num_epochs = self.config['training']['training_config']['num_epochs']
        patience = self.config['training']['training_config']['early_stopping']['patience']
        min_delta = self.config['training']['training_config']['early_stopping']['min_delta']
        
        # Training stages for MoE
        if self.config['model']['architecture'] == 'dcn_v2_moe' and self.config['model']['moe_config']['enabled']:
            stage_name = "Stage 1: Training Global Expert"
            print(f"\n{stage_name}...")
            global_expert_epochs = self.config['model']['moe_config']['training_stages']['global_expert_epochs']
            self.model.set_moe_training_stage('global')
            best_val_metric = self._train_stage(train_loader, val_loader, global_expert_epochs, stage_name)
            
            stage_name = "Stage 2: Training Regional Experts"
            print(f"\n{stage_name}...")
            regional_expert_epochs = self.config['model']['moe_config']['training_stages']['regional_expert_epochs']
            self.model.set_moe_training_stage('regional')
            best_val_metric = min(best_val_metric, self._train_stage(train_loader, val_loader, regional_expert_epochs, stage_name))
        else:
            best_val_metric = self._train_stage(train_loader, val_loader, num_epochs)
        
        return best_val_metric

    def _train_stage(self, train_loader: DataLoader, val_loader: Optional[DataLoader], num_epochs: int, stage_name: str = "") -> float:
        """Train for a specific stage (used for both regular training and MoE stages)."""
        best_val_metric = float('inf')
        patience_counter = 0
        min_delta = self.config['training']['training_config']['early_stopping']['min_delta']
        patience = self.config['training']['training_config']['early_stopping']['patience']
        
        for epoch in range(num_epochs):
            prefix = f"{stage_name} " if stage_name else ""
            print(f"\n{prefix}Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['total_loss']
                
                # Early stopping check
                if val_loss < best_val_metric - min_delta:
                    best_val_metric = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"\nBest validation metric: {best_val_metric:.4f} at epoch {epoch - patience}")
                    break
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        return best_val_metric

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary of metrics for each task
        """
        self.model.eval()
        all_outputs = {}
        all_labels = {}
        total_loss = 0.0
        num_batches = 0
        
        # Get enabled tasks
        tasks = [task for task, spec in self.config['tasks'].items() if spec['enabled']]
        
        # Initialize outputs and labels
        for task in tasks:
            all_outputs[task] = []
            all_labels[task] = []
        
        with torch.no_grad():
            for batch_features, batch_labels in dataloader:
                # Move features to device
                features = {
                    name: value.to(self.device) if isinstance(value, torch.Tensor)
                    else value
                    for name, value in batch_features.items()
                }
                
                # Move labels to device
                labels = {
                    name: value.to(self.device)
                    for name, value in batch_labels.items()
                }
                
                # Get predictions
                outputs = self.model(features)
                
                # Compute loss
                loss = self.compute_loss(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
                
                # Collect outputs and labels
                for task in tasks:
                    if task in outputs and task in batch_labels:
                        all_outputs[task].append(outputs[task].cpu().numpy())
                        all_labels[task].append(batch_labels[task].cpu().numpy())
        
        # Concatenate predictions and labels
        for task in tasks:
            if len(all_outputs[task]) > 0:
                all_outputs[task] = np.concatenate(all_outputs[task])
                all_labels[task] = np.concatenate(all_labels[task])
        
        # Compute metrics
        metrics = {}
        for task in tasks:
            if len(all_outputs[task]) == 0:
                continue
                
            task_metrics = {}
            y_true = all_labels[task]
            y_pred = all_outputs[task]
            
            # AUC
            try:
                task_metrics['auc'] = roc_auc_score(y_true, y_pred)
            except:
                task_metrics['auc'] = 0.0
            
            # NDCG at different k
            for k in [5, 10, 20]:
                task_metrics[f'ndcg@{k}'] = compute_ndcg(y_true, y_pred, k=k)
            
            # MAP at different k
            for k in [5, 10, 20]:
                task_metrics[f'map@{k}'] = compute_map(y_true, y_pred, k=k)
            
            # Precision at different k
            for k in [1, 3, 5, 10]:
                task_metrics[f'p@{k}'] = compute_precision_at_k(y_true, y_pred, k=k)
            
            # Recall at different k
            for k in [5, 10, 20]:
                task_metrics[f'r@{k}'] = compute_recall_at_k(y_true, y_pred, k=k)
            
            metrics[task] = task_metrics
        
        # Add total loss
        metrics['total_loss'] = total_loss / num_batches
        
        return metrics

    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for all enabled tasks.
        
        Args:
            outputs: Dictionary of model outputs
            labels: Dictionary of ground truth labels
            
        Returns:
            Total loss
        """
        total_loss = 0.0
        
        # Compute loss for each task
        for task, spec in self.config['tasks'].items():
            if not spec['enabled']:
                continue
                
            if task not in outputs or task not in labels:
                continue
                
            # Get task-specific loss function
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([spec['pos_weight']], device=self.device)
            ) if spec.get('pos_weight') else nn.BCEWithLogitsLoss()
            
            # Compute task loss
            task_loss = loss_fn(outputs[task], labels[task])
            
            # Apply task weight
            task_weight = spec.get('weight', 1.0)
            total_loss += task_weight * task_loss
        
        return total_loss 