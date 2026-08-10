from typing import Optional

import pandas as pd

from pgse.environment.ray_env import RayEnvManager
from pgse.dataset.alphabet import AUTO, Alphabet, AlphabetArg, ComplementArg, set_alphabet
from pgse.log import logger
from pgse.model.model_trainer import ModelTrainer
from pgse.dataset.loader import Loader
from pgse.dataset.sample_source import SampleSource
from pgse.dataset.source_factory import build_source
from pgse.dataset.table_label import DEFAULT_LABEL_COLUMN, TableArg
from pgse.segment import seg_pool
from pgse.validation import Metric, check_binary_labels


class Pipeline:
    def __init__(
            self,
            data_dir: Optional[str] = None,
            label_file: Optional[str | dict] = None,
            pre_kfold_info_file: Optional[str] = None,
            export_file: str = './default.export',
            k: int = 6,
            ext: int = 2,
            folds: int = 0,
            ea_min: Optional[float] = None,
            ea_max: Optional[float] = None,
            num_rounds: int = 1500,
            lr: float = 0.03,
            dist: bool = False,
            nodes: int = 1,
            workers: int = 8,
            device: str = 'cpu',
            alphabet: AlphabetArg = None,
            case_sensitive: bool = False,
            complement: ComplementArg = AUTO,
            metric: Optional[str] = None,
            binary: bool = False,
            table_file: Optional[TableArg] = None,
            data_column: Optional[str] = None,
            label_column: str = DEFAULT_LABEL_COLUMN
    ) -> None:
        """
        :param table_file: CSV file, or DataFrame, holding one sample per row. Setting it
            switches the run to table mode: the sequences are read from data_column of
            the table instead of from the files under data_dir.
        :param data_column: Name of the column holding the sequence of each sample.
            Required in table mode.
        :param label_column: Name of the column holding the label of each sample. Read in
            table mode.
        """
        # Install the alphabet first: everything downstream reads it.
        self.alphabet: Alphabet = set_alphabet(alphabet, case_sensitive=case_sensitive, complement=complement)
        logger.info(f'Using {self.alphabet}')

        self.data_dir = data_dir
        self.label_file = label_file
        self.pre_kfold_info_file = pre_kfold_info_file
        self.table_file = table_file
        self.data_column = data_column
        self.label_column = label_column
        self.export_file = export_file
        self.k = k
        self.ext = ext
        self.folds = folds
        self.ea_min = ea_min
        self.ea_max = ea_max
        self.num_rounds = num_rounds
        self.lr = lr
        self.dist = dist
        self.nodes = nodes
        self.workers = workers
        self.device = device
        self.binary = binary
        self.metric = metric or Metric.default_for(binary)

        self.source: SampleSource = build_source(
            data_dir, label_file, pre_kfold_info_file, table_file, data_column, label_column
        )

    def run(self):
        RayEnvManager.initialize(self.dist, self.nodes, self.workers)

        accumulated_results = pd.DataFrame()

        # Use k-mer data only without any feature selection or partitioning
        for i in range(self.folds if self.folds > 0 else 1):
            logger.info(f'==================== Fold {i + 1} ====================')
            loader = Loader(
                self.source,
                folds=self.folds,
                fold_index=i,
                workers=self.workers,
                dist=self.dist,
                nodes=self.nodes
            )

            if self.binary:
                check_binary_labels(loader.train_labels, loader.test_labels)

            model_trainer = ModelTrainer(
                loader,
                self.num_rounds,
                self.workers,
                self.lr,
                ea_min=self.ea_min,
                ea_max=self.ea_max,
                device=self.device,
                metric=self.metric,
                binary=self.binary
            )

            # Load k-mer dataset
            seg_pool.clear()
            seg_pool.add_all_kmer(self.k, self.ext)
            train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool()

            # Run XGBoost without partitioning or custom metrics
            custom_metric = model_trainer.build_validation_metric()
            fold_results, importance_df, trained_model = model_trainer.run_xgboost(
                train_kmer, test_kmer, train_labels, test_labels, use_partition=False, custom_metric=custom_metric
            )

            logger.info(fold_results)
            logger.info("Feature importance:")
            logger.info(str(importance_df.head(20)))

            # Append fold results
            accumulated_results = pd.concat([accumulated_results, fold_results], ignore_index=True)
            trained_model.save_model(f'{self.export_file}_regular_xgboost_fold_{i}')


        # Export final results and shutdown Ray
        accumulated_results.to_csv(f'{self.export_file}_regular_xgboost.csv', index=False)
        RayEnvManager.shutdown()
