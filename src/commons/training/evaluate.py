import torch
import numpy as np
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from ..models.dcn import DCNv2

def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: Optional[int] = None) -> float:
    """Compute Normalized Discounted Cumulative Gain."""
    # Ensure arrays are 1D
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
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

def compute_map(y_true: np.ndarray, y_pred: np.ndarray, k: Optional[int] = None) -> float:
    """Compute Mean Average Precision."""
    # Ensure arrays are 1D
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    if k is None:
        return average_precision_score(y_true, y_pred)
    
    # Get top k predictions
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    y_true_k = y_true[top_k_indices]
    y_pred_k = y_pred[top_k_indices]
    
    return average_precision_score(y_true_k, y_pred_k)

def compute_recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute Recall@k."""
    # Ensure arrays are 1D
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    # Get top k indices
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    
    # Count relevant items in top k
    num_relevant_in_top_k = np.sum(y_true[top_k_indices])
    
    # Total number of relevant items
    total_relevant = np.sum(y_true)
    
    return num_relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0

def compute_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute Precision@k."""
    # Ensure arrays are 1D
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    # Get top k indices
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    
    # Calculate precision
    return np.mean(y_true[top_k_indices])

def evaluate_model(
    model: DCNv2,
    dataloader: DataLoader,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        config: Configuration dictionary
        device: Device to run evaluation on
    
    Returns:
        Dictionary of metrics for each task
    """
    model.eval()
    all_outputs = {}
    all_labels = {}
    
    # Get enabled tasks
    tasks = [task for task, spec in config['tasks'].items() if spec['enabled']]
    
    # Initialize outputs and labels
    for task in tasks:
        all_outputs[task] = []
        all_labels[task] = []
    
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            # Move features to device
            features = {
                name: value.to(device) if isinstance(value, torch.Tensor)
                else (tuple(v.to(device) for v in value) if isinstance(value, tuple)
                      else value)
                for name, value in batch_features.items()
            }
            
            # Get predictions
            outputs = model(features)
            
            # Collect outputs and labels
            for task in tasks:
                if task in outputs and task in batch_labels:
                    all_outputs[task].append(outputs[task].cpu().numpy())
                    all_labels[task].append(batch_labels[task].numpy())
    
    # Concatenate predictions and labels
    for task in tasks:
        if all_outputs[task]:
            all_outputs[task] = np.concatenate(all_outputs[task])
            all_labels[task] = np.concatenate(all_labels[task])
    
    # Compute metrics
    metrics = {}
    for task in tasks:
        if not all_outputs[task]:
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
    
    return metrics 