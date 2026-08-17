import os
from typing import Optional

import numpy as np
import pandas as pd

from pgse.environment.ray_env import RayEnvManager
from pgse.dataset.alphabet import AUTO, Alphabet, AlphabetArg, ComplementArg, set_alphabet
from pgse.log import logger
from pgse.model.label_scaler import LabelScaler
from pgse.model.model_trainer import ModelTrainer
from pgse.model.pgse_model import PGSEModel
from pgse.dataset.label_utils import LabelColumns
from pgse.dataset.loader import Loader
from pgse.dataset.sample_source import SampleSource
from pgse.dataset.source_factory import build_source
from pgse.dataset.table_label import TableArg
from pgse.pipeline.progress_manager import ProgressManager
from pgse.result.fold_result import FoldResult
from pgse.result.predictions import to_matrices
from pgse.result.segment_importance import SegmentImportance
from pgse.result.training_result import TrainingResult
from pgse.segment.extender import Extender
from pgse.segment import seg_pool
from pgse.validation import Metric, check_binary_labels

# How far the labels' spreads may differ before standardising them is worth suggesting.
SCALE_WARNING_RATIO = 10.0


class Pipeline:
    def __init__(
            self,
            data_dir: Optional[str] = None,
            label_file: Optional[str | dict] = None,
            pre_kfold_info_file: Optional[str] = None,
            save_file: Optional[str] = None,
            export_file: Optional[str] = None,
            k: int = 6,
            ext: int = 2,
            target: int = 70,
            features: int = 10000,
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
            uint16: bool = False,
            sparse: bool = False,
            partition_size_target: int = 5000,
            metric: Optional[str] = None,
            binary: bool = False,
            table_file: Optional[TableArg] = None,
            data_column: Optional[str] = None,
            label_columns: Optional[LabelColumns] = None,
            standardise_labels: bool = False
    ) -> None:
        """
        :param data_dir: Directory holding one sequence file per sample. Read in file
            mode, i.e. when table_file is not given.
        :param label_file: CSV file, or dict, pairing each file under data_dir with its
            labels. Read in file mode. Its sample files are always read from the 'files'
            column, and its labels from the columns named by label_columns.
        :param save_file: Where to store the segment pool between rounds so a run can be
            resumed. Nothing is written when it is left unset.
        :param export_file: Path prefix for the models, segments and results written after
            each fold. Nothing is written when it is left unset.
        :param alphabet: str or Alphabet: The characters the sequences are made of.
            Defaults to DNA ('atgc'). Pass e.g. 'abcdefghijklmnopqrstuvwxyz ' to run
            PGSE over plain text.
        :param case_sensitive: bool: Treat upper and lower case as distinct characters.
        :param complement: The complement used to canonicalise segments. Defaults to
            reverse complementing for DNA and to no canonicalisation for any other
            alphabet. Pass None to switch it off explicitly.
        :param uint16: Store segment counts as uint16 instead of float32, halving the
            count-matrix footprint. Lossless for counts up to 65535 (saturated above).
        :param sparse: Store the count matrix as a sparse CSR matrix. For short
            sequences (e.g. SMILES) the matrix is almost all zeros, so this saves
            orders of magnitude. The same setting must be used at predict time.
        :param partition_size_target: Target features per XGBoost partition during feature
            selection. Partitions are evenly sized, so the actual size lands between this and
            twice this. 0 or less trains a single partition over all features.
        :param metric: Name of the validation metric reported for the held-out fold. See
            ``pgse.validation.Metric.names()`` for the full list. Defaults to essential
            agreement, which reads ea_min and ea_max, or to auroc in binary mode.
        :param binary: Train a 0/1 classifier (``binary:logistic``) instead of a regressor.
            Every label has to be 0 or 1, and a prediction is the probability of the 1.
        :param table_file: CSV file, or DataFrame, holding one sample per row. Setting it
            switches the run to table mode: the sequences are read from data_column of
            the table instead of from the files under data_dir, and data_dir, label_file
            and pre_kfold_info_file are left unread.
        :param data_column: Name of the column holding the sequence of each sample.
            Required in table mode.
        :param label_columns: Name of the column holding the label of each sample, or the
            names of several such columns. Naming several trains one XGBoost output per
            label over a single shared set of segments, and every label is scored, and
            weighs on the feature selection, in its own right. Required in file mode, and
            defaults to 'labels' in table mode.
        :param standardise_labels: Train on labels shifted to zero mean and unit variance,
            measured on the training fold alone. The predictions are returned in the units
            of the dataset, so this only changes what the model optimises: it stops a label
            on a larger scale from dominating the eval metric and early stopping of a
            multi-label run. Not available in binary mode.
        """
        # Install the alphabet first: everything downstream reads it, including the
        # segment extender and the Ray workers.
        self.alphabet: Alphabet = set_alphabet(alphabet, case_sensitive=case_sensitive, complement=complement)
        logger.info(f'Using {self.alphabet}')

        self.data_dir = data_dir
        self.label_file = label_file
        self.pre_kfold_info_file = pre_kfold_info_file
        self.table_file = table_file
        self.data_column = data_column
        self.label_columns = label_columns
        self.save_file = save_file
        self.export_file = export_file
        self.k = k
        self.ext = ext
        self.target = target
        self.features = features
        self.folds = folds
        self.ea_min = ea_min
        self.ea_max = ea_max
        self.dist = dist
        self.nodes = nodes
        self.num_rounds = num_rounds
        self.lr = lr
        self.workers = workers
        self.device = device
        self.count_dtype = np.uint16 if uint16 else np.float32
        self.sparse = sparse
        self.partition_size_target = partition_size_target
        self.binary = binary
        self.metric = metric or Metric.default_for(binary)

        if standardise_labels and binary:
            raise ValueError('standardise_labels is for continuous labels, so it cannot be used with binary.')
        self.standardise_labels = standardise_labels

        self.source: SampleSource = build_source(
            data_dir, label_file, pre_kfold_info_file, table_file, data_column, label_columns
        )
        self._warn_about_label_scales()

        self.extender = Extender()
        self.progress_manager = ProgressManager(self.save_file, self.k, self.ext)
        self.model_trainer = None

        # Set to False to keep a run entirely in memory. train() does this for the
        # duration of the run it starts.
        self.write_outputs: bool = True
        self.fold_results: list[FoldResult] = []

    def _warn_about_label_scales(self) -> None:
        """Warn when the labels sit on scales too far apart to be trained on as they are.

        XGBoost starts every label from one shared intercept, the mean over all of them,
        so a label whose values are far from it spends the whole run climbing back rather
        than fitting its own signal.
        """
        if self.binary or self.standardise_labels or self.source.n_labels < 2:
            return

        spreads = self.source.labels.std(axis=0)
        smallest, largest = float(spreads.min()), float(spreads.max())
        if smallest <= 0.0 or largest / smallest < SCALE_WARNING_RATIO:
            return

        widest = self.source.label_names[int(np.argmax(spreads))]
        narrowest = self.source.label_names[int(np.argmin(spreads))]
        logger.warning(
            f'Label {widest!r} varies {largest / smallest:.0f} times as widely as {narrowest!r}. '
            f'Every label is boosted from one shared intercept, so {narrowest!r} may barely '
            f'train. Pass standardise_labels=True (--standardise-labels 1) to train them on a '
            f'common scale.'
        )

    def extend_segments(self):
        try:
            self.extender.extend_all_segs(self.ext)
        except ValueError:
            logger.error("No segments could be extended. Stopping.")
            return False

        return True

    def run(self) -> TrainingResult:
        """Train every fold, returning the models, segments and scores they produced."""
        RayEnvManager.initialize(self.dist, self.nodes, self.workers)

        start_fold, accumulated_results = self.progress_manager.load_fold_progress()
        validation_metric = Metric(self.metric, ea_min=self.ea_min, ea_max=self.ea_max)
        self.fold_results = []

        for i in range(start_fold, self.folds if self.folds > 0 else 1):
            logger.info(f'==================== Fold {i + 1} ====================')
            loader = Loader(
                self.source,
                folds=self.folds,
                fold_index=i,
                count_dtype=self.count_dtype,
                sparse=self.sparse,
                workers=self.workers,
                dist=self.dist,
                nodes=self.nodes
            )

            if self.binary:
                check_binary_labels(loader.train_labels, loader.test_labels)

            scaler = LabelScaler.fit(np.asarray(loader.train_labels)) if self.standardise_labels else None

            self.model_trainer = ModelTrainer(
                loader,
                self.num_rounds,
                self.workers,
                self.lr,
                self.features,
                self.ea_min,
                self.ea_max,
                device=self.device,
                partition_size_target=self.partition_size_target,
                metric=self.metric,
                binary=self.binary,
                scaler=scaler
            )

            train_kmer, test_kmer, train_labels, test_labels = self.progress_manager.load_round_progress(loader)

            while True:
                logger.info(f'==================== Feature Selection ====================')

                # Step 1: Run XGBoost for feature selection
                xgb_result = self.model_trainer.run_xgboost(train_kmer, test_kmer, train_labels, test_labels)
                self.model_trainer.perform_feature_selection(xgb_result)

                # Step 2: Attempt to extend segments
                if seg_pool.get_current_max_length() >= self.target or not self.extend_segments():
                    break

                if self.write_outputs:
                    self.progress_manager.save_round_progress()

                train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool()

            # Step 3: Train and test with selected segments
            logger.info(f'==================== Training & testing with selected segments ====================')
            train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool()

            # Run XGBoost with custom metric
            fold_results, importance_df, trained_model = self.model_trainer.run_xgboost(
                train_kmer, test_kmer, train_labels, test_labels,
                use_partition=False, custom_metric=validation_metric
            )

            logger.info(fold_results)

            # Append fold results
            accumulated_results = self.progress_manager.append_results(fold_results, accumulated_results)

            model = PGSEModel(
                trained_model,
                SegmentImportance.from_xgb_importance(seg_pool.get_copy(), importance_df),
                self.alphabet,
                count_dtype=self.count_dtype,
                sparse=self.sparse,
                workers=self.workers,
                binary=self.binary,
                label_names=self.source.label_names,
                scaler=scaler
            )
            label_scores, score = self._score_fold(validation_metric, fold_results)
            self._log_fold_score(i, validation_metric.name, label_scores, score)

            self.fold_results.append(
                FoldResult(i, model, fold_results, validation_metric.name, score, label_scores)
            )

            # Save progress after each fold
            if self.write_outputs:
                self.progress_manager.save_fold_progress(i + 1, accumulated_results)
                self._remove_save_file()

                if self.export_file:
                    model.save(f'{self.export_file}_fold_{i}')

        # Export final results and shutdown Ray
        if self.write_outputs and self.export_file:
            accumulated_results.to_csv(f'{self.export_file}.csv')
        RayEnvManager.shutdown()

        return TrainingResult(self.fold_results, self.metric, validation_metric.greater_is_better)

    def train(self) -> TrainingResult:
        """Train every fold in memory, writing nothing to disk."""
        write_outputs = self.write_outputs
        self.write_outputs = False
        try:
            return self.run()
        finally:
            self.write_outputs = write_outputs

    def _score_fold(
            self,
            validation_metric: Metric,
            fold_results: pd.DataFrame
    ) -> tuple[dict[str, float], float]:
        """Score every label of a held-out fold, and average the scores into one.

        Args:
            validation_metric: The metric the fold is scored with.
            fold_results: The fold's predictions, one column pair per label.
        """
        actual, predicted = to_matrices(fold_results, self.source.label_names)
        scores = validation_metric.score_each(actual, predicted)

        label_scores = dict(zip(self.source.label_names, scores))
        return label_scores, float(np.mean(scores))

    def _log_fold_score(
            self,
            fold_index: int,
            metric_name: str,
            label_scores: dict[str, float],
            score: float
    ) -> None:
        """Log the score of a fold, per label when it carries several.

        Args:
            fold_index: Zero-based position of the fold.
            metric_name: Name of the validation metric.
            label_scores: The metric's value on each label.
            score: The mean of the per-label scores.
        """
        if len(label_scores) > 1:
            per_label = '  '.join(f'{name}={value}' for name, value in label_scores.items())
            logger.info(f'Fold {fold_index + 1} {metric_name}: {per_label}')
            logger.info(f'Fold {fold_index + 1} mean {metric_name} over {len(label_scores)} labels: {score}')
            return

        logger.info(f'Fold {fold_index + 1} {metric_name}: {score}')

    def _remove_save_file(self) -> None:
        """Drop the resume point, now that the fold it belonged to is finished."""
        if not self.save_file:
            return

        try:
            os.remove(self.save_file)
        except FileNotFoundError as e:
            logger.error(e)
