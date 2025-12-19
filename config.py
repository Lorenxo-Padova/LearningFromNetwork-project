"""
Configuration file for PPI Link Prediction Pipeline
"""
import os

# ==================== PATH CONFIGURATION ====================
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
PPI_DATASET = os.path.join(DATA_DIR, 'PP-Pathways_ppi.csv')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== CROSS-VALIDATION CONFIGURATION ====================
N_FOLDS = 4  # Number of folds for cross-validation
RANDOM_STATE = 42
NEGATIVE_SAMPLING_RATIO = 1.0  # Ratio of negative samples to positive samples

# ==================== EMBEDDING CONFIGURATION ====================
EMBEDDING_DIM = 32
WALK_LENGTH = 10
NUM_WALKS = 20
WORKERS_COUNT = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

# DeepWalk specific
DEEPWALK_WINDOW_SIZE = 10

# Node2Vec specific
NODE2VEC_P = 1.0  # Return parameter
NODE2VEC_Q = 1.0  # In-out parameter
NODE2VEC_WINDOW_SIZE = 10

# Metapath2Vec configuration

METAPATHS = [
    ["protein", "protein"],
    ["gene", "disease", "gene"],
    ["gene", "compound", "disease", "compound", "gene"],
    ["compound", "gene", "compound"]
]

METAPATH_WALK_LENGTH = 40
METAPATH_WALKS_PER_NODE = 10
METAPATH_WINDOW_SIZE = 5

# ==================== CLASSIFIER CONFIGURATION ====================
CLASSIFIER_TYPE = 'logistic_regression'  # Options: 'logistic_regression', 'random_forest', 'svm'
CLASSIFIER_MAX_ITER = 1000

# ==================== BINARY OPERATORS ====================
# Available operators for combining node embeddings
BINARY_OPERATORS = {
    'hadamard': lambda u, v: u * v,
    'average': lambda u, v: (u + v) / 2,
    'l1': lambda u, v: abs(u - v),
    'l2': lambda u, v: (u - v) ** 2,
}

DEFAULT_BINARY_OPERATOR = 'hadamard'

# ==================== EVALUATION CONFIGURATION ====================
METRICS = ['auc_roc', 'accuracy', 'precision', 'recall', 'f1']
SAVE_PLOTS = True
VERBOSE = True