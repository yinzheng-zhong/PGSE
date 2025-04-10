import json
import os
import pandas as pd

from pgse.log import logger
from pgse.segment import seg_pool

# Constants
DEFAULT_FEATURES_PRINT_COUNT = 20


class ProgressManager:
    def __init__(
            self,
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
        self.save_file = save_file
        self.k = k
        self.ext = ext

        self.progress_file = self.save_file + '.progress' if self.save_file else None

    def save_round_progress(self):
        """
        Save progress to the save file for PGSE rounds.
        """
        if not self.save_file:
            return

        seg_pool.save(self.save_file)

    def load_round_progress(self, loader):
        """
        Load progress from the save file for PGSE rounds.
        """
        try:
            seg_pool.load(self.save_file)
        except FileNotFoundError:
            seg_pool.clear()
            seg_pool.add_all_kmer(self.k, self.ext)

        return loader.get_dataset_from_pool()

    def save_fold_progress(self, fold_index, results):
        if not self.save_file:
            return

        progress_data = {
            'fold_index': fold_index,
            'results': results.to_dict()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f)
        logger.info(f"Progress saved at fold {fold_index}.")

    def load_fold_progress(self):
        if self.progress_file and os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
                fold_index = progress_data['fold_index']
                results = pd.DataFrame.from_dict(progress_data['results'])
                logger.info(f"Resuming from fold {fold_index + 1}.")
                return fold_index, results
        else:
            logger.info("No previous progress found, starting from the first fold.")
            return 0, pd.DataFrame()

    def append_results(self, new_results, existing_results):
        if existing_results.empty:
            return new_results
        else:
            return pd.concat([existing_results, new_results], ignore_index=True)
