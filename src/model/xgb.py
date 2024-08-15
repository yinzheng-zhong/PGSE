import os

import numpy as np
import pandas as pd
import xgboost as xgb

from src.dataset.loader import Loader
from src.model.util import is_essential_agreement, essential_agreement_cus_metric
from src.log import logger


class XGBoost:
    def __init__(
            self,
            boost_rounds: int = 250,
    ):
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'nthread': 76,
            'tree_method': 'approx',  # Using histogram-based method
        }

        self.boost_rounds = boost_rounds

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

        # Create DMatrix for training and testing
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x, label=test_y)

        # Watchlist to observe the training and testing performance
        watchlist = [(dtrain, 'train'), (dtest, 'test')]

        logger.info('Training the model...')
        model = xgb.train(self.params, dtrain, self.boost_rounds, watchlist, custom_metric=essential_agreement_cus_metric)

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
