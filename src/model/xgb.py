import os

import numpy as np
import pandas as pd
import xgboost as xgb

import ray
from src.model.util import is_essential_agreement, essential_agreement_cus_metric
from src.log import logger


class XGBoost:
    def __init__(
            self,
            boost_rounds: int = 250,
            num_nodes=1,
            num_cpu_per_node=8,
    ):
        self.boost_rounds = boost_rounds
        self.num_nodes = num_nodes
        self.num_cpu_per_node = num_cpu_per_node

        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1
        }

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

        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x, label=test_y)

        # Watchlist to observe the training and testing performance
        watchlist = [(dtrain, 'train'), (dtest, 'test')]

        logger.info('Training the model...')

        model = xgb.train(self.params, dtrain, self.boost_rounds, watchlist,
                          custom_metric=essential_agreement_cus_metric,
                          verbose_eval=100)  # Only print every 10 boosts

        # Print the final evaluation results
        evals_result = model.eval(dtest)
        print(evals_result)

        # Predict using the trained model
        predictions = model.predict(dtest)

        # Create a dataframe with predictions and actual labels
        results_df = pd.DataFrame({
            'Prediction': predictions,
            'Actual': dtest.get_label(),
            'Essential Agreement': list(is_essential_agreement(dtest.get_label(), predictions))
        })

        # Get feature importance
        feature_importance = model.get_score(importance_type='gain')
        # Convert dictionary to pandas dataframe
        importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
        # Sort by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        return results_df, importance_df
