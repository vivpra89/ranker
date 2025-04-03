import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Dict, List
import logging
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

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
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Initialize loss function for each task
        self.loss_fns = {}
        for task in self.tasks:
            loss_type = config['tasks'][task]['loss'].lower()
            if loss_type == 'bce':
                self.loss_fns[task] = nn.BCELoss()  # Using BCELoss since we apply sigmoid in model
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
    
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
        """Save model checkpoint."""
        if not self.config['logging']['checkpointing']['save_best'] and not is_best:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
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
        
        # Remove old checkpoints if needed
        max_checkpoints = self.config['logging']['checkpointing']['max_checkpoints']
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if len(checkpoints) > max_checkpoints:
            for checkpoint in checkpoints[:-max_checkpoints]:
                checkpoint.unlink()
    
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
    
    def train(self, dataloader):
        """Train the model."""
        num_epochs = self.config['training']['training_config']['num_epochs']
        early_stopping = self.config['training']['training_config']['early_stopping']
        
        # Split data into train and validation sets
        total_size = len(dataloader.dataset)
        train_size = int(self.config['training']['training_config']['train_split'] * total_size)
        val_size = total_size - train_size  # Use remaining data for validation
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataloader.dataset,
            [train_size, val_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            collate_fn=dataloader.collate_fn
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=dataloader.batch_size,
            collate_fn=dataloader.collate_fn
        )
        
        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            self.log_metrics(train_metrics, epoch, prefix='train/')
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.log_metrics(val_metrics, epoch, prefix='val/')
            
            # Save checkpoint and check for early stopping
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
                if early_stopping > 0 and self.patience_counter >= early_stopping:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Regular checkpoint saving
            if (epoch + 1) % self.config['logging']['checkpointing']['save_frequency'] == 0:
                self.save_checkpoint(epoch, val_metrics)
        
        print("Training completed!") 