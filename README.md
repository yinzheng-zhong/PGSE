# Progressive Genome Segment Enhancement (PGSE)

## Overview

PGSE is an algorithm for predicting phenotypes from
whole genome sequencing (WGS) data. It was intiially developed for the prediction
of antimicrobial minimum inhibitory concentration (MIC) in bacterial strains.
PGSE has higher accuracy, lower memory consumption, and shorter runtime compared
to traditional $k$-mer based XGBoost models.
PGSE is also able to run on distributed systems.

## Contributors

Dr Yinzheng (William) Zhong, Univerisity of Liverpool (algorithm design & implementation)

Dr Alessandro Gerada, University of Liverpool (conceptualisation, R package, funding)

Prof William Hope, University of Liverpool (conceptualisation, funding, supervision)

## Citation
```
@article{gerada2026prediction,
  title={Prediction of antimicrobial minimum inhibitory concentration from bacterial genomes using a scalable and interpretable machine learning approach},
  author={Gerada, Alessandro and Zhong, Yinzheng and Harper, Nicholas and Velluva, Anoop and Reza, Nada and Dubey, Vineet and Howard, Alex and Green, Peter L and Paterson, Steve and Hope, William},
  journal={npj Antimicrobials and Resistance},
  year={2026},
  publisher={Nature Publishing Group}
}
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.

## Installation

### PyPI

Make sure Python 3.10 or later is installed, then install `pgse` from PyPI with
[uv](https://docs.astral.sh/uv/):

```bash
uv pip install pgse
```

or with pip:

```bash
pip install pgse
```

The published wheels bundle the compiled native counting kernel (a Rust extension)
for Linux, macOS, and Windows.
Segment counting runs through this kernel; if it is ever unavailable (e.g. a source
install without a Rust toolchain) PGSE falls back to a slower pure-Python counter.

### Conda

To use in a conda environment:

```bash
conda create -n pgse python=3.11
conda activate pgse
python -m pip install pgse
```

`pgse` is now available to import.

### R

To use PGSE through R, install the package in an R session using:

```r
install.packages("devtools")
devtools::install_github("yinzheng-zhong/PGSE", subdir = "R-package")
```

## Usage

PGSE can be used either as a library, from your own Python program, or as a standalone
command-line program. Both do the same work; the sections below give each in turn, for
training and then for prediction.

### Training

#### As a library

`TrainingPipeline.train()` runs the whole pipeline in memory and hands back everything
it produced. It writes nothing: no models, no segment lists, no results, no log files
and no progress files, whatever paths were passed to the constructor. Use it when PGSE
is one step inside a larger program.

```python
from pgse import TrainingPipeline

result = TrainingPipeline(
    data_dir='genomes/',
    label_file='labels.csv',
    folds=5,
    metric='r2',
).train()

# 1. A predictive model, ready to use
model = result.model                      # the best-scoring fold; result.models has them all
predictions = model.predict(['new_1.fna', 'new_2.fna'])
predictions = model.predict_sequences(['ATGCATTACA...'])   # or straight from memory
features = model.count(files=['new_1.fna'])                # the raw segment-count matrix

# 2. The discovered segments and how important each one was
result.segments.top(10).to_frame()        # Segment / Importance, most important first
result.segments.segments                  # the segments alone, in count-matrix order
result.segments.importances               # the matching scores as a NumPy array

# 3. Per-fold detail
result.scores                             # the metric for each fold
result.predictions                        # held-out Prediction/Actual/Fold for every fold
result.to_frame()                         # one row per fold: score and segment count
for fold in result.folds:
    print(fold.index, fold.score, len(fold.segments))
```

`data_dir` and `label_file` read one sequence file per sample. To train from the rows of a
single CSV instead — one column holding the text, another the label — pass `table_file` and
`data_column`; see [Table mode](#table-mode).

A model carries its own segments, alphabet and count settings, so nothing has to be
passed alongside it and several models can be held at once. Predicting uses neither Ray
nor any global state. Saving is explicit, and everything needed to reload is written:

```python
from pgse import PGSEModel

model.save('artifacts/ecoli-caz')     # ecoli-caz.json, ecoli-caz_segs.csv, ecoli-caz_meta.json
reloaded = PGSEModel.load('artifacts/ecoli-caz')
reloaded.predict(['new_1.fna'])       # the alphabet and count settings come back with it
```

`run()` is the same run with its output written out: use it when you want the artefacts
on disk but are still driving PGSE from Python. Each fold's model, segments and metadata
go under `export_file`, and a resume point to `save_file`; whichever of the two is left
unset is simply not written.

```python
result = TrainingPipeline(
    data_dir='genomes/',
    label_file='labels.csv',
    save_file='run.save',         # resume point, removed once a fold finishes
    export_file='out/ecoli-caz',  # out/ecoli-caz_fold_0.json, _segs.csv, _meta.json, ...
    folds=5,
).run()
```

Those per-fold files are exactly what `save()` writes, so `PGSEModel.load('out/ecoli-caz_fold_0')`
picks up anything a `run()` or a `pgse-train` produced.

#### As a standalone program

To run PGSE as a standalone program on a local machine, install the package and use the following command as an example:
```bash
pgse-train \
        --label-file "../<path_to>/<you_labels>.csv" \
        --data-dir "../<you_data_dir>/" \
        --pre-kfold-info-file "../<k_fold_information>.json" \
        --save-file "../<saved progress>.save" \
        --export-file "../<exported files>" \
        --workers 8 \
        --features 10000 \
        --partition-size-target 5000 \
        --dist 0 \
        --k 6 \
        --target 70 \
        --ext 2 \
        --lr 0.001 \
        --num-rounds 6000 \
        --folds 5 \
        --ea-max 64 \
        --ea-min 0
```
* `--label-file` (Required): path to the .csv label file

    Here the label file is a csv file with the following format:
    ```text
    | labels | files     |
    | ------ | --------- |
    | 7      | file1.fna |
    | 7      | file2.fna |
    | 6      | file3.fna |
    ```

    The labels are the target values for the prediction task. The files are the file names (.fna files under `--data-dir`) containing the genome sequences.
* `--data-dir` (Required): path to the data directory containing the .fna files. PGSE will be able to retrieve the genome sequences using this path and the
file names in the label file.
* `--table-file`: path to a CSV file holding one sample per row. It replaces `--label-file` and
`--data-dir`: the sequences are read from a column of this file rather than from one file per
sample. See [Table mode](#table-mode) below.
* `--data-column`: name of the column of `--table-file` holding the sequence of each sample.
Required in table mode.
* `--label-column`: name of the column of `--table-file` holding the label of each sample
(default `labels`).
* `--pre-kfold-info-file`: path to the predefined k-fold info JSON file.
This is not required but will be useful if you want to compare PGSE with other systems. Without
this, PGSE will split the data into k folds randomly using a fixed seed. E.g.
    ```json
    {
        "fold_0": [
            "Sample_208-MOLMIC_E33.scaffolds.fna",
            "Sample_726-MOLMIC_F29.scaffolds.fna",
            "Sample_474-MOLMIC_I14.scaffolds.fna",
            "Sample_111-MOLMIC_C61.scaffolds.fna",
            "Sample_087-MOLMIC_C25.scaffolds.fna",
            "Sample_467-MOLMIC_I6.scaffolds.fna",
            "..."
        ],
        "fold_1": [
            "Sample_208-MOLMIC_E33.scaffolds.fna",
            "Sample_726-MOLMIC_F29.scaffolds.fna",
            "Sample_474-MOLMIC_I14.scaffolds.fna",
            "Sample_111-MOLMIC_C61.scaffolds.fna",
            "Sample_087-MOLMIC_C25.scaffolds.fna",
            "Sample_467-MOLMIC_I6.scaffolds.fna",
            "..."
        ],
        "...": [
        "..."
        ]
    }
    ```
* `--save-file`: file to save the progress. This is useful if you want to resume the training process.
Nothing is written when it is omitted.
* `--export-file`: path prefix for the results. Normally without an extension. Each fold writes
`<prefix>_fold_<i>.json` (the trained model), `<prefix>_fold_<i>_segs.csv` (the selected segments and
their importance) and `<prefix>_fold_<i>_meta.json` (the alphabet and count settings needed to
reload it), plus `<prefix>.csv` with the predictions of every fold. Nothing is written when it is
omitted.
* `--log-file`: file to append the log to. The log goes to the console only when it is omitted.
* `--workers`: number of workers per node.
* `--features`: Maximum number of features to keep after the feature importance calculation and ranking.
* `--partition-size-target`: Target number of features in each XGBoost partition during feature
selection (default `5000`). Partitions are evenly sized, so the actual size lands between this and
twice this. See [Why do we perform feature partitioning?](#why-do-we-perform-feature-partitioning)
below. Larger partitions use more memory per worker and give each model a wider view of the
features; `0` or less trains a single model over all features.
* `--dist`: Using distributed computation or not. 0 for running on a single node/machine, 1 for running on multiple nodes.
* `--k`: initial k-mer size.
* `--target`: Maximum segment length to extend to.
* `--ext`: Extension length in each round. Extension parameter `p` from the paper.
* `--lr`: learning rate.
* `--num-rounds`: Maximum rounds for the training process.
* `--folds`: Number of folds for the k-fold cross-validation.
* `--binary`: `0` (default) predicts a continuous value; `1` predicts a 0/1 label. See
[Binary mode](#binary-mode) below.
* `--metric`: Validation metric reported for the held-out fold (default `essential_agreement`, or
`auroc` with `--binary 1`). See [Validation metrics](#validation-metrics) below.
* `--ea-max`: Maximum number of censored essential agreement values. Don't need this unless
you want to see more accurate EA information from the console output during the training. Only read
by the `essential_agreement` metric.
* `--ea-min`: Minimum number of censored essential agreement values. Similar to `--ea-max`.
* `--alphabet`: the set of characters the input is made of, given as a single string. Defaults to
DNA (`atgc`). See [Alphabets](#alphabets) below.
* `--case-sensitive`: `0` (default) folds everything to lower case, `1` treats upper and lower case
as distinct characters.
* `--complement`: the complement used to canonicalise segments, one character per character of
`--alphabet`. See [Alphabets](#alphabets) below.
* `--uint16`: `0` (default) stores the segment-count matrix as `float32`; `1` stores it as `uint16`,
halving the memory. Lossless for counts up to 65535 (larger counts are saturated). See
[Reducing memory usage](#reducing-memory-usage) below.
* `--sparse`: `0` (default) stores the count matrix densely; `1` stores it as a sparse CSR matrix.
For short, sparse inputs (e.g. SMILES strings) the matrix is almost entirely zeros, so this can save
orders of magnitude. XGBoost reads the unstored zeros of a CSR matrix as *missing* values (not as
`0`), so the **same value must be used at prediction time**. See
[Reducing memory usage](#reducing-memory-usage) below.

Every flag above matches a `TrainingPipeline` argument of the same name, so anything the
command line can do the library can do too. `main-pgse.py` shows the two joined up: it
builds the pipeline from `pgse.environment.args.get_parser()`, the parser `pgse-train`
itself uses, which you can reuse or replace with your own.

#### Alphabets

PGSE is not limited to DNA. The alphabet is the set of characters the input is made of, and
everything outside it is dropped while reading the input files. Any symbolic text can therefore be
used, for example plain text:

```bash
pgse-train \
        --label-file "../<path_to>/<you_labels>.csv" \
        --data-dir "../<you_data_dir>/" \
        --alphabet "abcdefghijklmnopqrstuvwxyz " \
        --case-sensitive 0 \
        --k 3 \
        --target 12
```

The same options are available on the Python API:

```python
from pgse import TrainingPipeline

pipeline = TrainingPipeline(
    data_dir='...',
    label_file='...',
    k=3,
    target=12,
    alphabet='abcdefghijklmnopqrstuvwxyz ',
    case_sensitive=False
)
```

Three things follow from the alphabet:

* **Case sensitivity.** By default the alphabet is case-insensitive: the input is folded to lower
case, so `A` and `a` are the same character. With `--case-sensitive 1` nothing is folded and the
alphabet has to list every character you want to keep, e.g.
`"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "`.
* **Canonicalisation.** DNA segments are counted on both strands, so a segment and its reverse
complement are the same feature. This only makes sense when the alphabet has a complement.
`--complement` defaults to reverse complementing for the DNA alphabet and to no canonicalisation for
any other alphabet, which is almost always what you want. To set one explicitly, give one character
per character of `--alphabet` (`--alphabet augc --complement uacg` for RNA), or pass an empty string
to switch it off.
* **k-mer size.** The number of initial k-mers is `len(alphabet) ** k`, so a larger alphabet needs a
smaller `--k`. The 27-character alphabet above yields about 20k 3-mers, roughly the same as DNA with
`--k 7`.

Inference has to use the same alphabet as training, since the exported segments are counted against
it. Pass the same `--alphabet`, `--case-sensitive` and `--complement` values to `pgse-predict`;
if the segments do not fit the alphabet, PGSE fails with an error rather than predicting on
all-zero counts.

#### Table mode

For short text strings, **Table mode** reads them from a single
CSV instead: one row per sample, one column holding the sequence and another holding the label.

```text
| Compound_ID   | SMILES                           | active |
| ------------- | -------------------------------- | ------ |
| BRD-K28024289 | Nc1nnc(o1)-c1ccc(o1)[N+](=O)[O-] | TRUE   |
| BRD-K00556640 | O[C@H]1COC[C@@H]2O[C@H](CC...    | FALSE  |
```

```bash
pgse-train \
        --table-file "../<path_to>/<your_data>.csv" \
        --data-column "SMILES" \
        --label-column "active" \
        --alphabet "#()+-./0123456789=@BCFHINOPS[\]blnorsuc" \
        --case-sensitive 1 \
        --binary 1 \
        --sparse 1 \
        --k 2 \
        --target 8
```

`--table-file` replaces `--label-file` and `--data-dir`, which are not read in table mode, and any
column other than the two named is ignored. Everything else — folds, alphabets, binary mode,
metrics, feature selection, `--sparse` — behaves exactly as it does with one file per sample.

The Python API takes the same three arguments, and `table_file` also accepts a DataFrame that is
already in memory:

```python
import pandas as pd
from pgse import TrainingPipeline

frame = pd.read_csv('inhibition.csv')

result = TrainingPipeline(
    table_file=frame,
    data_column='SMILES',
    label_column='active',
    # the characters the column is made of; see Alphabets above
    alphabet=''.join(sorted(set(''.join(frame['SMILES'])))),
    case_sensitive=True,
    binary=True,
    sparse=True,
    k=2,
    target=8,
    folds=5,
).train()
```

Labels can be numbers, numeric strings or booleans (`TRUE`/`FALSE` is read as 1/0). Rows whose
sequence or label is empty are dropped with a warning, and a label that is not a number fails with
an error naming the offending values.

Prediction reads a table the same way. The exported CSV then holds the sequence column and the
prediction rather than a file name:

```bash
pgse-predict \
        --model-file "../<path_to_model>.json" \
        --segments-file "../<path_to_segments>.csv" \
        --table-file "../<new_compounds>.csv" \
        --data-column "SMILES" \
        --export-file "./predictions"
```

A table row carries its sequence with it, so the samples are read in process rather than through
the one Ray task per file that reading genomes needs.

#### Binary mode

By default PGSE predicts a continuous value. `--binary 1` (or `binary=True` on the Python API)
predicts a 0/1 label instead: XGBoost is trained with the `binary:logistic` objective, and a
prediction is the **probability that the sample is a 1**.

```bash
pgse-train \
        --label-file "../<path_to>/<your_labels>.csv" \
        --data-dir "../<your_data_dir>/" \
        --binary 1
```

```python
from pgse import TrainingPipeline

result = TrainingPipeline(
    data_dir='genomes/',
    label_file='labels.csv',
    folds=5,
    binary=True,
).train()

result.score                                # mean AUROC across the folds
probabilities = result.model.predict(['new_1.fna', 'new_2.fna'])   # e.g. [0.02, 0.91]
labels = (probabilities >= 0.5).astype(int)                        # if you want a decision
```

Every label has to be `0` or `1`; anything else fails with an error naming the offending values
rather than training a meaningless model. Booleans count as 0 and 1, so a `label_file` dictionary
of `True`/`False` works as well as one of `0`/`1`.

The label file is otherwise unchanged:

```text
| labels | files     |
| ------ | --------- |
| 1      | file1.fna |
| 0      | file2.fna |
| 1      | file3.fna |
```

Binary mode changes the objective, the default metric (`auroc` instead of `essential_agreement`)
and the meaning of a prediction. Everything else — segment extension, feature selection, folds,
alphabets, the saved model — behaves exactly as it does for a continuous target. A saved model
records that it is binary, so `PGSEModel.load` knows its predictions are probabilities:

```python
from pgse import PGSEModel

model = PGSEModel.load('artifacts/resistance')
model.binary       # True
model.predict(['new_1.fna'])   # a probability, not a class
```

#### Validation metrics

The score reported for the held-out fold is chosen with `--metric` (or `metric=` on the Python
API). Essential agreement is the default and keeps the behaviour of earlier versions, but it only
makes sense for log2 MIC labels; anything else should pick a metric that matches its label. A
binary run defaults to `auroc` instead.

| `--metric` | What it measures | Best for |
| --- | --- | --- |
| `essential_agreement` | Fraction of predictions within one two-fold dilution of the label | log2 MIC labels (default) |
| `rmse` | Root mean squared error, in the units of the label | Continuous labels, penalising large misses |
| `mae` | Mean absolute error | Continuous labels with outliers |
| `mape` | Mean absolute percentage error over the non-zero labels | Continuous labels compared in relative terms |
| `r2` | Fraction of the label's variance explained | Continuous labels, e.g. growth rate |
| `pearson` | Linear correlation between label and prediction | Continuous labels, e.g. growth rate |
| `spearman` | Rank correlation | Continuous labels where only the ordering matters |
| `accuracy` | Fraction of predictions that round to the exact label | Integer labels, e.g. 0/1 |
| `mcc` | Matthews correlation coefficient, in `[-1, 1]` | Imbalanced integer labels |
| `auroc` | Area under the ROC curve: the chance a positive outranks a negative | Binary labels (default with `--binary 1`) |
| `auprc` | Area under the precision-recall curve, which ignores the true negatives | Binary labels where the positives are rare |
| `log_loss` | Mean negative log likelihood, scoring the probabilities themselves | Binary labels where the probability matters, not just the ranking |
| `f1` | Harmonic mean of precision and recall | Binary labels, one headline number for the decision |
| `precision` | Fraction of the predicted positives that are positive | Binary labels where a false positive is costly |
| `recall` | Fraction of the positives that are predicted positive (sensitivity) | Binary labels where a missed positive is costly |
| `specificity` | Fraction of the negatives that are predicted negative | Binary labels, reported alongside recall |
| `balanced_accuracy` | Mean of recall and specificity | Imbalanced binary labels |

The bottom eight are for [binary mode](#binary-mode). `auroc`, `auprc` and `log_loss` read the
predicted probability directly; the rest need a yes/no answer and split the probability at `0.5`,
as `accuracy` and `mcc` already do. `log_loss` is the only one of them where a **smaller** score is
better, which `TrainingResult.best_fold` accounts for.

For a continuous target such as growth rate, `r2` is the usual headline number, with `rmse` for the
error in the target's own units and `spearman` when only the ranking of the samples matters:

```bash
pgse-train \
        --label-file "../<path_to>/<your_labels>.csv" \
        --data-dir "../<your_data_dir>/" \
        --metric r2
```

```python
from pgse import TrainingPipeline

pipeline = TrainingPipeline(data_dir='...', label_file='...', metric='r2')
```

PGSE always logs RMSE; `--metric` selects the extra score reported alongside it once the selected
segments are trained.

The metric does not change what PGSE optimises. The objective is set by `--binary` alone:
`reg:squarederror` by default and `binary:logistic` with `--binary 1`. So `--metric auroc` on a
continuous run scores the regressor's output as if it were a ranking, which is legal but rarely
what you want — pass `--binary 1` to actually train a classifier. `accuracy` and `mcc` work either
way, on whole-number labels: both round the predictions to the nearest integer before scoring.

The metrics are the static methods of the `Metric` class in `pgse/validation/metrics.py`, with the
array handling they share in `pgse/validation/utils.py`. Adding a metric is adding a method — its
name becomes the `--metric` value and the first line of its docstring becomes its entry in
`--metric`'s help:

```python
class Metric:
    ...

    @staticmethod
    def max_error(y_true, y_pred):
        """Largest absolute error over the samples.

        Args:
            y_true: True labels.
            y_pred: Predicted values.
        """
        return float(np.max(np.abs(as_array(y_true) - as_array(y_pred))))
```

A metric that needs extra parameters just declares them as keyword arguments. The pipeline passes
all of its validation options to every metric and each receives only the ones its signature names,
which is how `essential_agreement` gets `ea_min` and `ea_max` while the others ignore them.

#### Reducing memory usage

The segment-count matrix (one row per sample, one column per segment) is the largest object PGSE
holds in memory. Two optional flags shrink it; they are independent and can be combined. Both default
to off, so existing runs are unaffected.

* `--uint16 1` stores counts as 16-bit integers instead of 32-bit floats, halving the matrix. Counts
are non-negative integers, so this is lossless up to 65535; any larger count is saturated to 65535.
* `--sparse 1` stores the matrix as a sparse CSR matrix. This is the big lever for **short inputs
where most segments are absent from most samples** — for example SMILES strings, where each row has
only tens of non-zero counts out of thousands or millions of columns, making the matrix >99% zeros.
For long, dense inputs such as bacterial genomes the matrix is not sparse, so leave this off.

```bash
pgse-train \
        --label-file "../<path_to>/<your_labels>.csv" \
        --data-dir "../<your_data_dir>/" \
        --alphabet "..." \
        --sparse 1 \
        --uint16 1
```

The same options are available on the Python API as `sparse=True` and `uint16=True`:

```python
from pgse import TrainingPipeline

pipeline = TrainingPipeline(data_dir='...', label_file='...', sparse=True, uint16=True)
```

> **Important:** `--sparse` must match between training and prediction. In a sparse matrix, unstored
> zeros are read by XGBoost as *missing* values, whereas in a dense matrix a count of zero is an
> explicit `0`. Training dense and predicting sparse (or vice versa) shifts the predictions. `--uint16`
> has no such constraint. Pass the same values to `pgse-predict` (see below).

#### Distributed computation

To run PGSE on a distributed system, you need to use
your environment specific setup. There are multiple examples about running PGSE
using Slurm under the slurm-scripts directory.
* `job-pgse-array.sh`: Run PGSE on a cluster using Slurm with multiple nodes for multiple antibiotics using array jobs.
Here `-dist` is set to 0 as each task is running separately.
* `job-pgse-dist.sh`: Run PGSE on a cluster using Slurm with multiple nodes for a single antibiotic.
Here `-dist` is set to 1 as the task is running on different nodes.
* `job-pgse-single.sh`: Run PGSE on a Slurm cluster with a single node for a single antibiotic.
Here `-dist` is set to 0.

### Prediction

#### As a library

`PGSEModel` is the whole interface: load one, predict with it, keep it around for as many
calls as you like.

```python
from pgse import PGSEModel

model = PGSEModel.load('out/ecoli-caz_fold_0')
model.predict(['sample_1.fna', 'sample_2.fna'])
```

The alphabet and the count settings are stored with the model and restored on load, so
there is nothing to keep in step by hand. See [As a library](#as-a-library) above for the
rest of what a model can do.

`InferencePipeline` is the older interface, kept for existing callers: it takes the model
and segment files separately, and the alphabet and `sparse` setting have to be passed
again by hand to match the ones used for training. `main-pgse-inf.py` is a worked example.

```python
from pgse import InferencePipeline

pipeline = InferencePipeline(
    '../volatile/var/result-k6-CAZ-perf_fold_0.json',
    '../volatile/var/result-k6-CAZ-perf_fold_0_segs.csv',
    workers=8
)
print(pipeline.run(['../volatile/cgr/Sample_002-MOLMIC_B2.scaffolds.fna']))
```

#### As a standalone program

To run prediction as a standalone program, install the package and use the following command as an example:
```shell
pgse-predict \
        --model-file "../<path_to_model>.json" \
        --segments-file "../<path_to_segments>.csv" \
        --data-dir "../<you_data_dir>/" \
        --workers 8
```

If the model was trained on a non-DNA alphabet, pass the same `--alphabet`, `--case-sensitive` and
`--complement` values that were used for training. See [Alphabets](#alphabets). If training used
`--sparse 1`, prediction must use it too — see [Reducing memory usage](#reducing-memory-usage).

To score the rows of a CSV instead of a directory of files, pass `--table-file` and `--data-column`
in place of `--data-dir`. See [Table mode](#table-mode).

### Logging

Logging goes to the console only. Set `PGSE_LOG_FILE`, pass `--log-file` on the command
line, or call `pgse.log.logger.add_file_handler(path)`, to also append it to a file.

### R package

To use PGSE through the R package, consult the package
[documentation](https://github.com/yinzheng-zhong/PGSE/tree/main/R-package/).

## For Development

PGSE uses [uv](https://docs.astral.sh/uv/) for packaging and dependency
management. All project metadata and dependencies live in `pyproject.toml`.

**Prerequisite: a Rust toolchain.** Building from source compiles the native
counting kernel (the Rust crate under `native/`, built into the package as
`pgse._native`), so you need `cargo`/`rustc` on your `PATH`. Install them from
[rustup.rs](https://rustup.rs):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# rustup only updates PATH for *new* shells, so load it into the current one
# (or open a new shell) before installing:
. "$HOME/.cargo/env"
```
The build fails loudly if the toolchain is missing rather than silently shipping an
install without the extension, so every install has the fast path. (A pure-Python
counter still exists as a runtime safety net, and logs a warning if it is ever used.)

Clone the repository and create a synced environment. This installs the runtime
and `dev` dependencies and performs an editable install of `pgse`, compiling the
Rust kernel as part of it:
```bash
uv sync
```

To run an editable install on its own:
```bash
uv pip install -e .
```

To build the distribution artifacts (the wheel bundles the compiled kernel):
```bash
uv build
```

Run the tests with:
```bash
uv run pytest
```

Releases to PyPI are automated by GitHub Actions: when the `version` in
`pyproject.toml` changes on `main`, the workflow uses
[cibuildwheel](https://cibuildwheel.pypa.io/) to build one abi3 wheel per platform
(Linux, macOS, Windows; each valid for CPython 3.10+), builds an sdist, and
publishes them.

## Acknowledgements

This work was funded, in part, by UKRI and the Wellcome trust.

This work was undertaken on Barkla, part of the High Performance Computing
facilities at the Univeristy of Liverpool, UK.

## Common Issues
### XGBoost training is only using one core.
Some linux distributions need an environment variable `OMP_NUM_THREADS=<num threads>` to be set to allow XGBoost to use multiple cores. 

## Q & A

### Why do we perform feature partitioning?

There are four reasons why feature partitioning is crucial in PGSE. 
First, feature partitioning is used as a memory reduction technique.
The model is trained on a subset of the features at a time, therefore, the memory consumption is reduced while maintained
a relatively stable RAM usage regardless of the number of total features.
Second, feature partitioning helps to parallelise the training process. Each partition can be trained on a different worker
across different nodes. This is particularly useful as XGBoost training consumes most of the time in the training process.
Third, from the experiments we have conducted, we found that feature dimensionality affects the model's optimal hyperparameters.
For example, higher feature dimensionality requires a shallower tree depth in general.
PGSE is a dynamic system that and the total number of features can be different in each round.
Therefore, partitioning the features into similarly-sized sub-features can help to minimise the impact of the feature dimensionality on the model's hyperparameters.
Finally, feature partitioning helps to preserve the feature importance information from XGBoost. Likely due to the
pruning process, more feature importance information will be lost (become 0) if the dimensionality increases.

The size of each partition is set with `--partition-size-target` (or `partition_size_target=` on
`TrainingPipeline`), which defaults to 5000 features.

### Why do we eliminate features?

If segment `A` is extended into segment `B`, `A` becomes a subsequences of `B`. For pairs like `A` and `B`, we only need to keep the
ones with higher feature importance. Extension and elimination are two crucial parts of the PGSE system, which grows the
genome segments longer and the elimination process guarantees that the growth will stop eventually. Additionally, elimination
guarantees the convergence of the system as the feature dimensionality will start decreasing at some point till
all features stop growing.
