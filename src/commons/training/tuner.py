import optuna
from typing import Dict, Any
import copy
import logging
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from ..models.dcn import DCNv2
from .trainer import ReRankerTrainer
from ..data.collate import collate_fn

class HyperparameterTuner:
    def __init__(self, base_config: Dict[str, Any], dataset, tuning_config: Dict[str, Any]):
        self.base_config = base_config
        self.dataset = dataset
        self.tuning_config = tuning_config
        self.best_trial = None
        self.best_score = float('-inf')
        self.study = None
        self.setup_logging()
        
    def setup_logging(self):
        log_dir = Path(self.tuning_config['checkpointing']['checkpoint_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_dir / f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def create_trial_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Create a config for this trial by sampling from parameter space."""
        config = copy.deepcopy(self.base_config)
        
        for param_path, param_config in self.tuning_config['parameter_space'].items():
            # Parse the parameter path (e.g., "model.dcn_config.cross_layers")
            path_parts = param_path.split('.')
            target = config
            for part in path_parts[:-1]:
                target = target[part]
            
            # Sample parameter based on type
            if param_config['type'] == 'int':
                value = trial.suggest_int(
                    param_path,
                    param_config['range'][0],
                    param_config['range'][1]
                )
            elif param_config['type'] == 'float':
                if param_config.get('log', False):
                    value = trial.suggest_float(
                        param_path,
                        param_config['range'][0],
                        param_config['range'][1],
                        log=True
                    )
                else:
                    value = trial.suggest_float(
                        param_path,
                        param_config['range'][0],
                        param_config['range'][1]
                    )
            elif param_config['type'] == 'categorical':
                value = trial.suggest_categorical(
                    param_path,
                    param_config['values']
                )
            
            # Update config with sampled value
            target[path_parts[-1]] = value
        
        return config
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for a single trial."""
        # Create config for this trial
        trial_config = self.create_trial_config(trial)
        
        # Setup cross-validation
        if self.tuning_config['cross_validation']['stratify']:
            # Use first label for stratification
            first_label = next(iter(self.dataset.labels.values()))
            kf = StratifiedKFold(
                n_splits=self.tuning_config['cross_validation']['n_splits'],
                shuffle=self.tuning_config['cross_validation']['shuffle']
            )
            splits = kf.split(
                np.zeros(len(self.dataset)),
                first_label.numpy().argmax(axis=1) if len(first_label.shape) > 1 else first_label.numpy()
            )
        else:
            kf = KFold(
                n_splits=self.tuning_config['cross_validation']['n_splits'],
                shuffle=self.tuning_config['cross_validation']['shuffle']
            )
            splits = kf.split(np.zeros(len(self.dataset)))
        
        # Cross-validation loop
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logging.info(f"Trial {trial.number} Fold {fold_idx + 1}/{self.tuning_config['cross_validation']['n_splits']}")
            
            # Create train/val datasets
            train_subset = torch.utils.data.Subset(self.dataset, train_idx)
            val_subset = torch.utils.data.Subset(self.dataset, val_idx)
            
            # Create dataloaders
            train_loader = DataLoader(
                train_subset,
                batch_size=trial_config['training']['training_config']['batch_size'],
                shuffle=True,
                num_workers=0,
                collate_fn=lambda batch: collate_fn(batch, trial_config)
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=trial_config['training']['training_config']['batch_size'],
                shuffle=False,
                num_workers=0,
                collate_fn=lambda batch: collate_fn(batch, trial_config)
            )
            
            # Initialize model, optimizer, and scheduler
            model = DCNv2(trial_config)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=trial_config['training']['optimizer']['learning_rate'],
                weight_decay=trial_config['training']['optimizer']['weight_decay']
            )
            
            # Initialize trainer
            trainer = ReRankerTrainer(
                model=model,
                config=trial_config,
                optimizer=optimizer
            )
            
            # Train and evaluate
            best_score = trainer.train(
                train_loader,
                val_loader=val_loader,
                early_stopping_patience=self.tuning_config['early_stopping']['patience'],
                early_stopping_min_delta=self.tuning_config['early_stopping']['min_delta']
            )
            
            fold_scores.append(best_score)
            
            # Report intermediate value
            trial.report(best_score, fold_idx)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Calculate final score (mean across folds)
        final_score = np.mean(fold_scores)
        
        # Update best trial if needed
        if final_score > self.best_score:
            self.best_score = final_score
            self.best_trial = {
                'number': trial.number,
                'params': trial.params,
                'score': final_score,
                'config': trial_config
            }
            
            # Save best trial
            self.save_best_trial()
        
        return final_score
    
    def save_best_trial(self):
        """Save the best trial configuration and results."""
        save_dir = Path(self.tuning_config['checkpointing']['checkpoint_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / 'best_trial.json', 'w') as f:
            json.dump(self.best_trial, f, indent=2)
    
    def tune(self):
        """Run the hyperparameter tuning process."""
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            self.objective,
            n_trials=self.tuning_config['strategy']['n_trials'],
            timeout=self.tuning_config['strategy']['timeout_hours'] * 3600 if self.tuning_config['strategy']['timeout_hours'] else None,
            n_jobs=self.tuning_config['strategy']['n_jobs']
        )
        
        self.study = study
        return self.best_trial 