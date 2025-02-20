import math

import pandas as pd
import xgboost

from pgse.log import logger
from pgse.model.util import essential_agreement_cus_metric
from pgse.model.xgb import XGBoost
from pgse.segment import seg_pool

# Constants
DEFAULT_PARTITION_SIZE = 5000
DEFAULT_FEATURES_PRINT_COUNT = 20


class ModelTrainer:
    def __init__(
            self,
            loader,
            num_rounds=1500,
            workers=8,
            lr=0.03,
            features=10000,
            ea_min=None,
            ea_max=None
    ):
        self.loader = loader
        self.num_rounds = num_rounds
        self.workers = workers
        self.lr = lr
        self.features = features
        self.ea_min = ea_min
        self.ea_max = ea_max

    def run_xgboost(
            self,
            train_kmer,
            test_kmer,
            train_labels,
            test_labels,
            use_partition=True,
            custom_metric=None
    ) -> tuple[pd.DataFrame, pd.DataFrame, xgboost.Booster]:
        xgb = XGBoost(
            partition_size=DEFAULT_PARTITION_SIZE,
            boost_rounds=self.num_rounds,
            num_cpu_per_node=self.workers,
            use_partition=use_partition,
            base_learning_rate=self.lr,
            custom_metric=custom_metric,
            early_stopping_rounds=20
        )
        return xgb.run(train_kmer, test_kmer, train_labels, test_labels)

    def perform_feature_selection(self, xgb_result):
        _, importance_df, _ = xgb_result
        logger.info(str(importance_df.head(DEFAULT_FEATURES_PRINT_COUNT)))

        # Select top features
        index = list(map(int, importance_df['Feature'].values))[:self.features]
        seg_pool.use_subset(index)
        seg_pool.redundant_elimination(range(len(index)))

    def custom_essential_agreement_metric(self):
        return lambda x, y: essential_agreement_cus_metric(
            x, y,
            min_after_log2=math.log2(self.ea_min) if self.ea_min else None,
            max_after_log2=math.log2(self.ea_max) if self.ea_max else None
        )