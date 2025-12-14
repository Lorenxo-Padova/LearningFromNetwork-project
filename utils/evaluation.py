"""
Evaluation utilities for link prediction
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_feature_vectors(edge_df, embeddings, binary_operator):
    """
    Create feature vectors for edges using node embeddings and binary operator.
    
    Args:
        edge_df (pandas.DataFrame): DataFrame with 'Source' and 'Target' columns
        embeddings (dict): Dictionary mapping nodes to embedding vectors
        binary_operator (callable): Function to combine two embedding vectors
        
    Returns:
        tuple: (X, valid_indices) where X is feature matrix and valid_indices are rows with valid embeddings
    """
    X = []
    valid_indices = []
    
    for idx, row in edge_df.iterrows():
        u = str(row['Source'])
        v = str(row['Target'])
        
        # Check if both nodes have embeddings
        if u in embeddings and v in embeddings:
            u_emb = embeddings[u]
            v_emb = embeddings[v]
            
            # Apply binary operator
            feature_vector = binary_operator(u_emb, v_emb)
            X.append(feature_vector)
            valid_indices.append(idx)
    
    return np.array(X), valid_indices

def evaluate_predictions(y_true, y_pred, y_pred_proba):
    """
    Calculate various evaluation metrics.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_pred_proba (array): Predicted probabilities
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics

def print_metrics(metrics, fold=None):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics (dict): Dictionary of metrics
        fold (int, optional): Fold number for cross-validation
    """
    if fold is not None:
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1} Results")
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print(f"Overall Results")
        print(f"{'='*50}")
    
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}\n")

def aggregate_cv_results(all_fold_metrics):
    """
    Aggregate results from all cross-validation folds.
    
    Args:
        all_fold_metrics (list): List of metric dictionaries from each fold
        
    Returns:
        dict: Dictionary with mean and std for each metric
    """
    aggregated = {}
    
    for metric_name in all_fold_metrics[0].keys():
        values = [fold[metric_name] for fold in all_fold_metrics]
        aggregated[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    return aggregated

def print_cv_summary(aggregated_metrics):
    """
    Print summary of cross-validation results.
    
    Args:
        aggregated_metrics (dict): Aggregated metrics from all folds
    """
    print(f"\n{'='*50}")
    print(f"Cross-Validation Summary (Mean ± Std)")
    print(f"{'='*50}")
    
    for metric_name, stats in aggregated_metrics.items():
        print(f"{metric_name.upper():10s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

def plot_roc_curves(y_true_list, y_pred_proba_list, save_path=None):
    """
    Plot ROC curves for all folds.
    
    Args:
        y_true_list (list): List of true labels for each fold
        y_pred_proba_list (list): List of predicted probabilities for each fold
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for fold, (y_true, y_pred_proba) in enumerate(zip(y_true_list, y_pred_proba_list)):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        plt.plot(fpr, tpr, label=f'Fold {fold + 1} (AUC = {auc:.3f})', alpha=0.7)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Cross-Validation')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        save_path (str, optional): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def save_results_to_file(results, filepath):
    """
    Save evaluation results to a text file.
    
    Args:
        results (dict): Results dictionary
        filepath (str): Path to save the file
    """
    with open(filepath, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Link Prediction Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        # Write configuration
        if 'config' in results:
            f.write("Configuration:\n")
            f.write("-"*60 + "\n")
            for key, value in results['config'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        # Write per-fold results
        if 'fold_results' in results:
            f.write("Per-Fold Results:\n")
            f.write("-"*60 + "\n")
            for fold_idx, fold_metrics in enumerate(results['fold_results']):
                f.write(f"\nFold {fold_idx + 1}:\n")
                for metric, value in fold_metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
        
        # Write aggregated results
        if 'aggregated' in results:
            f.write("\n" + "="*60 + "\n")
            f.write("Cross-Validation Summary:\n")
            f.write("-"*60 + "\n")
            for metric, stats in results['aggregated'].items():
                f.write(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")