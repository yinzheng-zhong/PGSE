import math
import os
import ray
import pandas as pd

from src.log import logger
from src.model.util import essential_agreement_cus_metric
from src.enviromnet import args
from src.dataset.file_label import FileLabel
from src.dataset.loader import Loader
from src.segment.extender import Extender
from src.model.xgb import XGBoost
from src.segment import seg_pool

# Constants for easier adjustments
DEFAULT_PARTITION_SIZE = 5000
DEFAULT_FEATURES_PRINT_COUNT = 20


def initialize_environment():
    """Initialize environment variables and Ray."""
    os.environ["RAY_LOG_TO_STDERR"] = "0"
    os.environ["RAY_LOG_LEVEL"] = "ERROR"

    if args.dist:
        ray.init(address='auto', log_to_driver=True)
        logger.warning(
            f'Connected to Ray cluster with {args.nodes} nodes and {args.workers} workers per node.\n'
            f'Sometimes the progress bar may seem frozen, but it is still running.'
        )
    else:
        ray.init(num_cpus=args.workers, log_to_driver=True)


def load_progress(loader):
    """Load datasets, either from saved file or create new if not found."""
    try:
        seg_pool.load(args.save_file)
        return loader.get_dataset_from_pool(no_consecutive=False)
    except FileNotFoundError:
        seg_pool.add_all_kmer(args.k, args.ext)
        return loader.get_kmer_dataset(args.k, no_consecutive=False)


def run_xgboost(train_kmer, test_kmer, train_labels, test_labels, use_partition=True, custom_metric=None):
    """Initialize and run XGBoost with provided parameters."""
    xgb = XGBoost(
        partition_size=DEFAULT_PARTITION_SIZE,
        boost_rounds=args.num_rounds,
        num_cpu_per_node=args.workers,
        use_partition=use_partition,
        base_learning_rate=args.lr,
        custom_metric=custom_metric
    )
    return xgb.run(train_kmer, test_kmer, train_labels, test_labels)


def perform_feature_selection(xgb_result):
    """Select the top features based on importance."""
    _, importance_df = xgb_result
    logger.info(str(importance_df.head(DEFAULT_FEATURES_PRINT_COUNT)))

    # Select top features
    index = list(map(int, importance_df['Feature'].values))[:args.features]
    seg_pool.use_subset(index)
    seg_pool.redundant_elimination(range(len(index)))


def extend_segments(extender):
    """Try to extend segments and handle errors if any."""
    try:
        extender.extend_all_segs(args.ext)
    except ValueError:  # If no segments can be extended
        logger.error("No segments could be extended. Stopping.")
        return False
    return True


def custom_essential_agreement_metric():
    """Custom metric for essential agreement in XGBoost."""
    return lambda x, y: essential_agreement_cus_metric(
        x, y,
        min_after_log2=math.log2(args.ea_min) if args.ea_min is not None else None,
        max_after_log2=math.log2(args.ea_max) if args.ea_max is not None else None
    )


def main():
    """Main workflow for the process."""
    # Initialize environment and dependencies
    initialize_environment()

    file_label = FileLabel(args.label_file, args.data_dir)
    loader = Loader(file_label, folds=args.folds, fold_index=args.fold_index)
    extender = Extender()

    # Load initial datasets
    train_kmer, test_kmer, train_labels, test_labels = load_progress(loader)

    # Iterative process for feature selection and model training
    while True:
        logger.info(f'==================== Feature Selection ====================')

        # Step 1: Run XGBoost for feature selection
        xgb_result = run_xgboost(train_kmer, test_kmer, train_labels, test_labels)
        perform_feature_selection(xgb_result)

        # Step 2: Train and test with selected segments
        logger.info(f'==================== Training & testing with selected segments ====================')
        train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool(no_consecutive=False)

        # Run XGBoost with custom metric
        custom_metric = custom_essential_agreement_metric()
        results, _ = run_xgboost(train_kmer, test_kmer, train_labels, test_labels, use_partition=False,
                                 custom_metric=custom_metric)

        # Step 3: Attempt to extend segments
        if seg_pool.get_current_max_length() >= args.target or not extend_segments(extender):
            break

        seg_pool.save(args.save_file)
        train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool(no_consecutive=False)

    # Export results and shutdown Ray
    results.to_csv(f'{args.export_file}.csv')
    seg_pool.export(args.export_file)
    ray.shutdown()


if __name__ == "__main__":
    main()
