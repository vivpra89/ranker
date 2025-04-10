import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Optional, Tuple

def setup_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: Dict
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """Setup optimizer and learning rate scheduler.
    
    Args:
        model: The model to optimize
        config: Configuration dictionary containing optimizer and scheduler settings
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer_config = config['training']['optimizer']
    scheduler_config = config['training']['scheduler']
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_config['learning_rate'],
        weight_decay=optimizer_config['weight_decay'],
        betas=(optimizer_config['beta1'], optimizer_config['beta2'])
    )
    
    # Setup scheduler
    if scheduler_config['enabled']:
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=scheduler_config['warmup_steps']
        )
        
        # Main scheduler
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['training_config']['num_epochs'] - scheduler_config['warmup_steps'],
            eta_min=scheduler_config['min_lr']
        )
        
        # Combine schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[scheduler_config['warmup_steps']]
        )
    else:
        scheduler = None
    
    return optimizer, scheduler 