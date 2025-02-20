import json
import os
import pandas as pd

from src.log import logger
from src.segment import seg_pool

# Constants
DEFAULT_FEATURES_PRINT_COUNT = 20


class ProgressManager:
    def __init__(
            self,
            loader,
            save_file: str,
            k: int,
            ext: int,
    ):
        """
        ProgressManager constructor.
        @param loader: Loader object.
        @param save_file: File to save progress.
        @param k: Initial K-mer size.
        @param ext: Extension size (p).
        """
        self.loader = loader
        self.save_file = save_file
        self.k = k
        self.ext = ext

    def load_progress(self):
        try:
            seg_pool.load(self.save_file)
        except FileNotFoundError:
            seg_pool.clear()
            seg_pool.add_all_kmer(self.k, self.ext)

        return self.loader.get_dataset_from_pool()

    def save_fold_progress(self, fold_index, results, progress_file):
        progress_data = {
            'fold_index': fold_index,
            'results': results.to_dict()
        }
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f)
        logger.info(f"Progress saved at fold {fold_index}.")

    @staticmethod
    def load_fold_progress(progress_file):
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                fold_index = progress_data['fold_index']
                results = pd.DataFrame.from_dict(progress_data['results'])
                logger.info(f"Resuming from fold {fold_index + 1}.")
                return fold_index, results
        else:
            logger.info("No previous progress found, starting from the first fold.")
            return 0, pd.DataFrame()

    @staticmethod
    def append_results(new_results, existing_results):
        if existing_results.empty:
            return new_results
        else:
            return pd.concat([existing_results, new_results], ignore_index=True)
