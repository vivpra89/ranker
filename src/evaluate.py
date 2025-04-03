import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from models.dcn import DCNv2
from utils.config import load_config
from inference import ReRankerInference
import logging
import argparse
from pathlib import Path
from sklearn.metrics import ndcg_score, average_precision_score
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReRankerEvaluator:
    def __init__(self, reranker: ReRankerInference):
        """
        Initialize the reranker evaluator.
        
        Args:
            reranker: Initialized ReRankerInference instance
        """
        self.reranker = reranker
        self.config = reranker.config
    
    def compute_ranking_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = None
    ) -> Dict[str, float]:
        """
        Compute ranking metrics.
        
        Args:
            y_true: Ground truth relevance scores
            y_pred: Predicted relevance scores
            k: Optional cutoff for top-k metrics
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Reshape for sklearn metrics
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)
        
        # NDCG@k
        if k is not None:
            metrics[f'ndcg@{k}'] = ndcg_score(y_true, y_pred, k=k)
        else:
            metrics['ndcg'] = ndcg_score(y_true, y_pred)
        
        # Mean Average Precision
        metrics['map'] = average_precision_score(y_true.squeeze() > 0, y_pred.squeeze())
        
        # Mean Reciprocal Rank (MRR)
        ranked_relevance = y_true.squeeze()[np.argsort(-y_pred.squeeze())]
        first_relevant = np.where(ranked_relevance > 0)[0]
        if len(first_relevant) > 0:
            metrics['mrr'] = 1.0 / (first_relevant[0] + 1)
        else:
            metrics['mrr'] = 0.0
        
        return metrics
    
    def evaluate_task(
        self,
        user_features: Dict[str, Any],
        product_features: List[Dict[str, Any]],
        ground_truth: Dict[str, np.ndarray],
        task: str,
        k: int = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance for a specific task.
        
        Args:
            user_features: Dictionary of user features
            product_features: List of product feature dictionaries
            ground_truth: Dictionary of ground truth labels for each task
            task: Task name to evaluate
            k: Optional cutoff for top-k metrics
            
        Returns:
            Dictionary of metric scores
        """
        # Get predictions
        predictions = self.reranker.predict_scores(user_features, product_features)
        task_preds = predictions[task]
        
        # Compute metrics
        return self.compute_ranking_metrics(
            ground_truth[task],
            task_preds,
            k=k
        )
    
    def analyze_feature_importance(
        self,
        user_features: Dict[str, Any],
        product_features: List[Dict[str, Any]],
        feature_name: str,
        num_perturbations: int = 10
    ) -> Dict[str, float]:
        """
        Analyze feature importance through perturbation analysis.
        
        Args:
            user_features: Dictionary of user features
            product_features: List of product feature dictionaries
            feature_name: Name of feature to analyze
            num_perturbations: Number of random perturbations
            
        Returns:
            Dictionary of impact scores for each task
        """
        # Get baseline predictions
        baseline_preds = self.reranker.predict_scores(user_features, product_features)
        
        # Initialize impact scores
        impact_scores = defaultdict(list)
        
        for _ in range(num_perturbations):
            # Create perturbed features
            if feature_name in user_features:
                perturbed_user = user_features.copy()
                if isinstance(user_features[feature_name], np.ndarray):
                    perturbed_user[feature_name] = np.random.randn(*user_features[feature_name].shape)
                perturbed_product = product_features
            else:
                perturbed_user = user_features
                perturbed_product = []
                for prod in product_features:
                    perturbed_prod = prod.copy()
                    if isinstance(prod[feature_name], np.ndarray):
                        perturbed_prod[feature_name] = np.random.randn(*prod[feature_name].shape)
                    perturbed_product.append(perturbed_prod)
            
            # Get perturbed predictions
            perturbed_preds = self.reranker.predict_scores(perturbed_user, perturbed_product)
            
            # Compute impact for each task
            for task in baseline_preds:
                impact = np.mean(np.abs(baseline_preds[task] - perturbed_preds[task]))
                impact_scores[task].append(impact)
        
        # Average impact scores
        return {
            task: np.mean(scores)
            for task, scores in impact_scores.items()
        }
    
    def analyze_task_correlations(
        self,
        user_features: Dict[str, Any],
        product_features: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Analyze correlations between task predictions.
        
        Args:
            user_features: Dictionary of user features
            product_features: List of product feature dictionaries
            
        Returns:
            DataFrame of task correlations
        """
        # Get predictions for all tasks
        predictions = self.reranker.predict_scores(user_features, product_features)
        
        # Create DataFrame of predictions
        df = pd.DataFrame(predictions)
        
        # Compute correlations
        return df.corr()
    
    def plot_score_distributions(
        self,
        user_features: Dict[str, Any],
        product_features: List[Dict[str, Any]],
        save_path: str = None
    ):
        """
        Plot distributions of predicted scores for each task.
        
        Args:
            user_features: Dictionary of user features
            product_features: List of product feature dictionaries
            save_path: Optional path to save the plot
        """
        predictions = self.reranker.predict_scores(user_features, product_features)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot distributions
        for task, scores in predictions.items():
            sns.kdeplot(scores, label=task)
        
        plt.title('Score Distributions by Task')
        plt.xlabel('Predicted Score')
        plt.ylabel('Density')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def generate_evaluation_report(
        self,
        test_data: Dict[str, Any],
        output_dir: Path,
        k_values: List[int] = [5, 10, 20]
    ):
        """
        Generate comprehensive evaluation report.
        
        Args:
            test_data: Dictionary containing test data
            output_dir: Directory to save evaluation results
            k_values: List of k values for top-k metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = defaultdict(dict)
        
        # Evaluate ranking metrics for each task and k
        for task in self.config['tasks']:
            if not self.config['tasks'][task]['enabled']:
                continue
                
            for k in k_values:
                metrics = self.evaluate_task(
                    test_data['user_features'],
                    test_data['product_features'],
                    test_data['ground_truth'],
                    task,
                    k=k
                )
                for metric, value in metrics.items():
                    results[task][metric] = value
        
        # Analyze feature importance
        feature_importance = {}
        for feature in self.config['features']['enabled_features']:
            if not self.config['features']['feature_configs'][feature]['enabled']:
                continue
                
            importance = self.analyze_feature_importance(
                test_data['user_features'],
                test_data['product_features'],
                feature
            )
            feature_importance[feature] = importance
        
        # Analyze task correlations
        task_correlations = self.analyze_task_correlations(
            test_data['user_features'],
            test_data['product_features']
        )
        
        # Plot score distributions
        self.plot_score_distributions(
            test_data['user_features'],
            test_data['product_features'],
            save_path=output_dir / 'score_distributions.png'
        )
        
        # Save results
        pd.DataFrame(results).to_csv(output_dir / 'ranking_metrics.csv')
        pd.DataFrame(feature_importance).to_csv(output_dir / 'feature_importance.csv')
        task_correlations.to_csv(output_dir / 'task_correlations.csv')
        
        # Generate summary report
        with open(output_dir / 'evaluation_summary.txt', 'w') as f:
            f.write("Reranker Evaluation Summary\n")
            f.write("==========================\n\n")
            
            f.write("Ranking Metrics:\n")
            f.write("--------------\n")
            for task, metrics in results.items():
                f.write(f"\n{task}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\nFeature Importance:\n")
            f.write("-----------------\n")
            for feature, importance in feature_importance.items():
                f.write(f"\n{feature}:\n")
                for task, value in importance.items():
                    f.write(f"  {task}: {value:.4f}\n")
            
            f.write("\nTask Correlations:\n")
            f.write("----------------\n")
            f.write(task_correlations.to_string())

def load_test_data(file_path: str) -> Dict[str, Any]:
    """Load test data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Helper function to convert numpy array format to actual numpy arrays
    def to_numpy(value, dtype=np.float32):
        if isinstance(value, dict) and value.get('__type__') == 'ndarray':
            return np.array(value['data'], dtype=dtype).reshape(value['shape'])
        return np.array(value, dtype=dtype)
    
    # Convert lists to numpy arrays
    data['user_features']['user'] = to_numpy(data['user_features']['user'])
    
    for product in data['product_features']:
        product['product'] = to_numpy(product['product'])
        product['price'] = to_numpy(product['price'])
        product['category'] = to_numpy(product['category'])
    
    for task in data['ground_truth']:
        data['ground_truth'][task] = to_numpy(data['ground_truth'][task])
    
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default=None, help="Device to run evaluation on")
    args = parser.parse_args()
    
    # Initialize reranker
    reranker = ReRankerInference(args.model_path, args.config_path, args.device)
    evaluator = ReRankerEvaluator(reranker)
    
    # Load test data
    test_data = load_test_data(args.test_data_path)
    
    # Generate evaluation report
    evaluator.generate_evaluation_report(
        test_data,
        args.output_dir,
        k_values=[1, 2, 3]  # Adjust k values based on test data size
    )
    
    logger.info(f"Evaluation results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 