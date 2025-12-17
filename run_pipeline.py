"""
Main pipeline for PPI Link Prediction with Cross-Validation
"""
import argparse
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import os

# Import configurations
import config

# Import utilities
from utils.data_loader import (
    load_ppi_data, create_cv_folds, prepare_fold_data
)
from utils.evaluation import (
    create_feature_vectors, evaluate_predictions, 
    print_metrics, aggregate_cv_results, print_cv_summary,
    plot_roc_curves, save_results_to_file
)

# Import embedding methods
from embeddings.deepwalk import DeepWalkEmbedder
from embeddings.node2vec import Node2VecEmbedder
from embeddings.graphwave.graphwave import GraphWaveEmbedder

# Available embedding methods
EMBEDDING_METHODS = {
    'deepwalk': DeepWalkEmbedder,
    'node2vec': Node2VecEmbedder,
    'graphwave': GraphWaveEmbedder,
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PPI Link Prediction Pipeline with Cross-Validation'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='deepwalk',
        choices=list(EMBEDDING_METHODS.keys()),
        help='Embedding method to use'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=config.PPI_DATASET,
        help='Path to PPI dataset'
    )
    
    parser.add_argument(
        '--folds',
        type=int,
        default=config.N_FOLDS,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=config.EMBEDDING_DIM,
        help='Embedding dimension'
    )
    
    parser.add_argument(
        '--binary-op',
        type=str,
        default=config.DEFAULT_BINARY_OPERATOR,
        choices=list(config.BINARY_OPERATORS.keys()),
        help='Binary operator for combining embeddings'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=config.RESULTS_DIR,
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=config.VERBOSE,
        help='Verbose output'
    )
    
    return parser.parse_args()

def run_single_fold(fold_idx, train_edges, test_pos_edges, test_neg_edges, 
                   train_graph, embedder, binary_operator, verbose=True):
    """
    Run link prediction for a single fold.
    
    Args:
        fold_idx (int): Fold index
        train_edges (DataFrame): Training edges
        test_pos_edges (DataFrame): Positive test edges
        test_neg_edges (DataFrame): Negative test edges
        train_graph (Graph): Training graph
        embedder: Embedding method instance
        binary_operator: Function to combine embeddings
        verbose (bool): Print progress
        
    Returns:
        tuple: (metrics, y_true, y_pred_proba)
    """
    if verbose:
        print(f"\n--- Fold {fold_idx + 1} ---")
        print(f"Training edges: {len(train_edges)}")
        print(f"Test positive edges: {len(test_pos_edges)}")
        print(f"Test negative edges: {len(test_neg_edges)}")
    
    # Generate embeddings on training graph
    if verbose:
        print("Generating embeddings...")
    embeddings = embedder.generate_embeddings(train_graph)
    
    # Prepare test data
    test_pos_edges['Label'] = 1
    test_neg_edges['Label'] = 0
    test_data = pd.concat([test_pos_edges, test_neg_edges], ignore_index=True)
    test_data = test_data.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    
    # Create feature vectors
    if verbose:
        print("Creating feature vectors...")
    X_test, valid_indices = create_feature_vectors(test_data, embeddings, binary_operator)
    y_test = test_data.loc[valid_indices, 'Label'].values
    
    if len(X_test) == 0:
        raise ValueError("No valid feature vectors created. Check if nodes in test set have embeddings.")
    
    # Split test data for training classifier
    split_point = len(X_test) // 2
    X_train_clf = X_test[:split_point]
    y_train_clf = y_test[:split_point]
    X_test_clf = X_test[split_point:]
    y_test_clf = y_test[split_point:]
    
    # Train classifier
    if verbose:
        print("Training classifier...")
    classifier = LogisticRegression(
        random_state=config.RANDOM_STATE,
        solver='lbfgs',
        max_iter=config.CLASSIFIER_MAX_ITER
    )
    classifier.fit(X_train_clf, y_train_clf)
    
    # Make predictions
    y_pred_proba = classifier.predict_proba(X_test_clf)[:, 1]
    y_pred = classifier.predict(X_test_clf)
    
    # Evaluate
    metrics = evaluate_predictions(y_test_clf, y_pred, y_pred_proba)
    
    if verbose:
        print_metrics(metrics, fold=fold_idx)
    
    return metrics, y_test_clf, y_pred_proba

def run_pipeline(args):
    """
    Run the complete link prediction pipeline with cross-validation.
    
    Args:
        args: Command line arguments
    """
    print("="*60)
    print("PPI Link Prediction Pipeline")
    print("="*60)
    
    start_time = time.time()
    
    # Load data
    print(f"\n1. Loading data from {args.data}...")
    edges_df = load_ppi_data(args.data)
    print(f"   Total edges loaded: {len(edges_df)}")
    
    # Create cross-validation folds
    print(f"\n2. Creating {args.folds}-fold cross-validation splits...")
    folds = create_cv_folds(edges_df, n_folds=args.folds, random_state=config.RANDOM_STATE)
    
    # Initialize embedding method
    print(f"\n3. Initializing {args.method} embedder...")
    if args.method == 'deepwalk':
        embedder = DeepWalkEmbedder(
            embedding_dim=args.embedding_dim,
            walk_length=config.WALK_LENGTH,
            num_walks=config.NUM_WALKS,
            window_size=config.DEEPWALK_WINDOW_SIZE,
            workers=config.WORKERS_COUNT,
            random_state=config.RANDOM_STATE
        )
    elif args.method == 'node2vec':
        embedder = Node2VecEmbedder(
            embedding_dim=args.embedding_dim,
            walk_length=config.WALK_LENGTH,
            num_walks=config.NUM_WALKS,
            p=config.NODE2VEC_P,
            q=config.NODE2VEC_Q,
            window_size=config.NODE2VEC_WINDOW_SIZE,
            workers=config.WORKERS_COUNT,
            random_state=config.RANDOM_STATE
        )
    elif args.method == 'graphwave':
        embedder = GraphWaveEmbedder(
            embedding_dim=args.embedding_dim,
            order=config.ORDER if hasattr(config, 'ORDER') else 30,
            proc=config.PROC if hasattr(config, 'PROC') else 'approximate',
            nb_filters=config.NB_FILTERS if hasattr(config, 'NB_FILTERS') else 2,
            random_state=config.RANDOM_STATE
        )
    
    print(f"   Embedder: {embedder}")
    
    # Get binary operator
    binary_operator = config.BINARY_OPERATORS[args.binary_op]
    print(f"   Binary operator: {args.binary_op}")
    
    # Run cross-validation
    print(f"\n4. Running {args.folds}-fold cross-validation...")
    all_fold_metrics = []
    y_true_all = []
    y_pred_proba_all = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(folds, desc="Folds")):
        # Prepare fold data
        print("Preparing fold data...")
        train_edges, test_pos, test_neg, train_graph = prepare_fold_data(
            edges_df, train_idx, test_idx, 
            negative_ratio=config.NEGATIVE_SAMPLING_RATIO,
            random_state=config.RANDOM_STATE + fold_idx
        )
        print("Runninng link prediction for this fold...")
        # Run prediction for this fold
        fold_metrics, y_true, y_pred_proba = run_single_fold(
            fold_idx, train_edges, test_pos, test_neg, train_graph,
            embedder, binary_operator, verbose=args.verbose
        )
        
        all_fold_metrics.append(fold_metrics)
        y_true_all.append(y_true)
        y_pred_proba_all.append(y_pred_proba)
    
    # Aggregate results
    print("\n5. Aggregating results...")
    aggregated_metrics = aggregate_cv_results(all_fold_metrics)
    print_cv_summary(aggregated_metrics)
    
    # Save results
    print(f"\n6. Saving results to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_prefix = f"{args.method}_{args.binary_op}_{timestamp}"
    
    results = {
        'config': {
            'method': args.method,
            'embedding_dim': args.embedding_dim,
            'n_folds': args.folds,
            'binary_operator': args.binary_op,
            'walk_length': config.WALK_LENGTH,
            'num_walks': config.NUM_WALKS,
        },
        'fold_results': all_fold_metrics,
        'aggregated': aggregated_metrics
    }
    
    # Save text results
    results_file = os.path.join(args.output_dir, f"{results_prefix}_results.txt")
    save_results_to_file(results, results_file)
    print(f"   Results saved to: {results_file}")
    
    # Plot ROC curves
    if config.SAVE_PLOTS:
        roc_plot_file = os.path.join(args.output_dir, f"{results_prefix}_roc_curves.png")
        plot_roc_curves(y_true_all, y_pred_proba_all, save_path=roc_plot_file)
        print(f"   ROC curves saved to: {roc_plot_file}")
    
    # Final summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline completed in {elapsed_time:.2f} seconds")
    print(f"{'='*60}")
    print(f"\nFinal AUC-ROC: {aggregated_metrics['auc_roc']['mean']:.4f} ± {aggregated_metrics['auc_roc']['std']:.4f}")

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)