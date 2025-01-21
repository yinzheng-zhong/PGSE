import math
from src.log import logger
from src.model.util import essential_agreement_cus_metric
from src.enviromnet import args
from src.model.xgb import XGBoost
from src.segment import seg_pool

# Constants
DEFAULT_PARTITION_SIZE = 5000
DEFAULT_FEATURES_PRINT_COUNT = 20
PROGRESS_FILE = args.save_file + '.progress'


class ModelTrainer:
    def __init__(self, loader):
        self.loader = loader

    def run_xgboost(self, train_kmer, test_kmer, train_labels, test_labels, use_partition=True, custom_metric=None):
        xgb = XGBoost(
            partition_size=DEFAULT_PARTITION_SIZE,
            boost_rounds=args.num_rounds,
            num_cpu_per_node=args.workers,
            use_partition=use_partition,
            base_learning_rate=args.lr,
            custom_metric=custom_metric,
            early_stopping_rounds=20
        )
        return xgb.run(train_kmer, test_kmer, train_labels, test_labels)

    @staticmethod
    def perform_feature_selection(xgb_result):
        _, importance_df = xgb_result
        logger.info(str(importance_df.head(DEFAULT_FEATURES_PRINT_COUNT)))

        # Select top features
        index = list(map(int, importance_df['Feature'].values))[:args.features]
        seg_pool.use_subset(index)
        seg_pool.redundant_elimination(range(len(index)))

    @staticmethod
    def custom_essential_agreement_metric():
        return lambda x, y: essential_agreement_cus_metric(
            x, y,
            min_after_log2=math.log2(args.ea_min) if args.ea_min else None,
            max_after_log2=math.log2(args.ea_max) if args.ea_max else None
        )