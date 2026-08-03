from typing import Optional

import pandas as pd
import xgboost

from pgse.dataset.loader import Loader
from pgse.log import logger
from pgse.model.xgb import XGBoost
from pgse.segment import seg_pool
from pgse.validation import Metric

# Constants
DEFAULT_FEATURES_PRINT_COUNT = 20


class ModelTrainer:
    def __init__(
            self,
            loader: Loader,
            num_rounds: int = 1500,
            workers: int = 8,
            lr: float = 0.03,
            features: int = 10000,
            ea_min: Optional[float] = None,
            ea_max: Optional[float] = None,
            device: str = 'cpu',
            partition_size_target: int = 5000,
            metric: str = Metric.DEFAULT
    ) -> None:
        self.loader = loader
        self.num_rounds = num_rounds
        self.workers = workers
        self.lr = lr
        self.features = features
        self.ea_min = ea_min
        self.ea_max = ea_max
        self.device = device
        self.partition_size_target = partition_size_target
        self.metric = metric

    def run_xgboost(
            self,
            train_kmer,
            test_kmer,
            train_labels,
            test_labels,
            use_partition: bool = True,
            custom_metric: Optional[Metric] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, xgboost.Booster]:
        xgb = XGBoost(
            partition_size_target=self.partition_size_target,
            boost_rounds=self.num_rounds,
            num_cpu_per_node=self.workers,
            use_partition=use_partition,
            base_learning_rate=self.lr,
            custom_metric=custom_metric,
            early_stopping_rounds=20,
            device=self.device
        )
        return xgb.run(train_kmer, test_kmer, train_labels, test_labels)

    def perform_feature_selection(self, xgb_result) -> None:
        _, importance_df, _ = xgb_result
        logger.info(str(importance_df.head(DEFAULT_FEATURES_PRINT_COUNT)))

        # Select top features
        index = list(map(int, importance_df['Feature'].values))[:self.features]
        seg_pool.use_subset(index)
        seg_pool.redundant_elimination(range(len(index)))

    def build_validation_metric(self) -> Metric:
        """The configured validation metric, with its parameters bound."""
        return Metric(self.metric, ea_min=self.ea_min, ea_max=self.ea_max)
