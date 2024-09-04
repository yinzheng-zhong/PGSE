import os
import sys
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import ray
from tqdm import tqdm

from src.model.util import is_essential_agreement, essential_agreement_cus_metric
from src.log import logger


class XGBoost:
    def __init__(
            self,
            boost_rounds: int = 250,
            max_depth: int = 4,
            learning_rate: float = 0.05,
            importance_type: str = 'gain',
            partition_size=0,
            num_cpu_per_node=8,
            custom_metric=None
    ):
        self.boost_rounds = boost_rounds
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.importance_type = importance_type
        self.partition_size = partition_size
        self.num_cpu_per_node = num_cpu_per_node
        self.custom_metric = custom_metric

        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'nthread': num_cpu_per_node  # Use multiple threads per worker
        }

    @ray.remote(num_cpus=1)
    def train_sub_features(
            self,
            train_x: np.ndarray,
            train_y: np.ndarray,
            test_x: np.ndarray,
            test_y: np.ndarray,
            feature_indices: np.ndarray,  # Pass the original feature indices
            verbose: int = 100
    ):
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x, label=test_y)

        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(
            self.params, dtrain, self.boost_rounds, watchlist,
            custom_metric=self.custom_metric,
            verbose_eval=verbose
        )

        # Predict using the trained model
        predictions = model.predict(dtest)

        # Create a dataframe with predictions and actual labels
        results = {
            'Prediction': predictions,
            'Actual': dtest.get_label()
        }

        importance = model.get_score(importance_type=self.importance_type,)

        # Map back to the original feature indices
        importance_mapped = {feature_indices[int(k[1:])]: v for k, v in importance.items()}

        return results, list(importance_mapped.items())

    def run(
            self,
            train_x: np.ndarray,
            test_x: np.ndarray,
            train_y: np.ndarray,
            test_y: np.ndarray,
    ):
        """
        Train and test the model using the initial parameters
        :param train_x:
        :param test_x:
        :param train_y:
        :param test_y:
        :return:
        """

        # Split the features across the available nodes
        num_splits = train_x.shape[1] // self.partition_size if self.partition_size > 0 else 1
        if num_splits <= 0:
            num_splits = 1
        feature_splits = np.array_split(np.arange(train_x.shape[1]), num_splits)

        # Store ray object references
        tasks = []

        logger.info(f'Training {num_splits} partitions of features')

        for split in feature_splits:
            train_x_split = train_x[:, split]
            test_x_split = test_x[:, split]

            # Assign each task to a node with the specified number of CPUs
            task_ref = self.train_sub_features.options(
                num_cpus=self.num_cpu_per_node
            ).remote(
                self,
                train_x_split,
                train_y,
                test_x_split,
                test_y,
                split,
                verbose=100 if not self.partition_size else 0
            )

            tasks.append(task_ref)

        # Gather the results from each node
        results = [ray.get(task) for task in tqdm(tasks, desc='Training partitions')]

        # Combine the predictions and models from each node
        combined_predictions = np.mean([res[0]['Prediction'] for res in results], axis=0)

        # Create a dataframe with combined results
        results_df = pd.DataFrame({
            'Prediction': combined_predictions,
            'Actual': test_y
        })

        # Concatenate the importance dataframes
        all_importance_dfs = []
        for res in results:
            all_importance_dfs += res[1]

        importance_df = pd.DataFrame(all_importance_dfs, columns=['Feature', 'Importance'], index=None)
        importance_df.sort_values(by='Importance', ascending=False, inplace=True)

        # rooted mean square error
        rmse = np.sqrt(np.mean((results_df['Prediction'] - results_df['Actual']) ** 2))
        logger.info(f'Rooted Mean Square Error: {rmse}')

        return results_df, importance_df
