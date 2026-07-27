# Progressive Genome Segment Enhancement (PGSE)

## Overview

PGSE is an algorithm for predicting phenotypes from
whole genome sequencing (WGS) data. It was intiially developed for the prediction
of antimicrobial minimum inhibitory concentration (MIC) in bacterial strains.
PGSE has higher accuracy, lower memory consumption, and shorter runtime compared
to traditional $k$-mer based XGBoost models.
PGSE is also able to run on distributed systems.

## Contributors

Dr Yinzheng (William) Zhong, Univerisity of Liverpool (algorithm design & Python implementation)

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

The published wheels bundle the prebuilt Aho-Corasick shared library for Linux,
macOS, and Windows, so no C compiler is required for a normal install.

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

### Training

#### Single node/machine

Import the pipeline from the package and run the pipeline like this.
You can use your own argument parser or use the one provided by pgse.
Also, you can instantiate the pipeline with a wrapper that provides the parameters directly.

```python
# You can use your own argument parser or use the one provided by pgse.
# Or instantiate the pipeline with a wrapper that provides the parameters directly.
from pgse.environment.args import get_parser
from pgse import TrainingPipeline

if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()

  pipeline = TrainingPipeline(
    args.data_dir,
    args.label_file,
    args.pre_kfold_info_file,
    args.save_file,
    args.export_file,
    args.k,
    args.ext,
    args.target,
    args.features,
    args.folds,
    args.ea_min,
    args.ea_max,
    args.num_rounds,
    args.lr,
    args.dist,
    args.nodes,
    args.workers
  )

  pipeline.run()
```


Alternatively, to run PGSE as a standalone program on a local machine, install the package and use the following command as an example:
```bash
pgse-train \
        --label-file "../<path_to>/<you_labels>.csv" \
        --data-dir "../<you_data_dir>/" \
        --pre-kfold-info-file "../<k_fold_information>.json" \
        --save-file "../<saved progress>.save" \
        --export-file "../<exported files>" \
        --workers 8 \
        --features 10000 \
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
* `--export-file`: file to export the results. Normally without an extension.
This name will be used to store the selected genome segments in an .txt file and the trained model in a .json file.
* `--workers`: number of workers per node.
* `--features`: Maximum number of features to keep after the feature importance calculation and ranking.
* `--dist`: Using distributed computation or not. 0 for running on a single node/machine, 1 for running on multiple nodes.
* `--k`: initial k-mer size.
* `--target`: Maximum segment length to extend to.
* `--ext`: Extension length in each round. Extension parameter `p` from the paper.
* `--lr`: learning rate.
* `--num-rounds`: Maximum rounds for the training process.
* `--folds`: Number of folds for the k-fold cross-validation.
* `--ea-max`: Maximum number of censored essential agreement values. Don't need this unless
you want to see more accurate EA information from the console output during the training.
* `--ea-min`: Minimum number of censored essential agreement values. Similar to `--ea-max`.
* `--alphabet`: the set of characters the input is made of, given as a single string. Defaults to
DNA (`atgc`). See [Alphabets](#alphabets) below.
* `--case-sensitive`: `0` (default) folds everything to lower case, `1` treats upper and lower case
as distinct characters.
* `--complement`: the complement used to canonicalise segments, one character per character of
`--alphabet`. See [Alphabets](#alphabets) below.

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

### Inferencing

An example of how this can be done is provided in `main-pgse-inf.py`.

```python
from pgse import InferencePipeline

MODEL_PATH = '../volatile/var/result-k6-CAZ-perf_fold_0.json'
SEGMENT_PATH = '../volatile/var/result-k6-CAZ-perf_fold_0.csv'

if __name__ == "__main__":
    # Instantiate the pipeline
    pipeline = InferencePipeline(MODEL_PATH, SEGMENT_PATH, workers=8)

    # files as a list of paths to the fasta files
    EG_1 = [
        '../volatile/cgr/Sample_002-MOLMIC_B2.scaffolds.fna',
        '../volatile/cgr/Sample_394-MOLMIC_H8.scaffolds.fna',
        '../volatile/cgr/Sample_385-MOLMIC_G79.scaffolds.fna',
        '../volatile/cgr/Sample_622-MOLMIC_K68.scaffolds.fna',
        '../volatile/cgr/Sample_252-MOLMIC_F2.scaffolds.fna',
        '../volatile/cgr/Sample_208-MOLMIC_E33.scaffolds.fna',
        '../volatile/cgr/Sample_443-MOLMIC_H62.scaffolds.fna',
        '../volatile/cgr/Sample_565-MOLMIC_J66.scaffolds.fna',
        '../volatile/cgr/Sample_339-MOLMIC_G29.scaffolds.fna',
        '../volatile/cgr/Sample_418-MOLMIC_H33.scaffolds.fna',
    ]

    result_1 = pipeline.run(EG_1)
    print(result_1)

    EG_2 = [
        '../volatile/cgr/Sample_394-MOLMIC_H8.scaffolds.fna',
        '../volatile/cgr/Sample_385-MOLMIC_G79.scaffolds.fna',
        '../volatile/cgr/Sample_622-MOLMIC_K68.scaffolds.fna',
        '../volatile/cgr/Sample_252-MOLMIC_F2.scaffolds.fna'
    ]

    result_2 = pipeline.run(EG_2)
    print(result_2)
```

To run the inference pipeline as a standalone program, install the package and use the following command as an example:
```shell
pgse-predict \
        --model-file "../<path_to_model>.json" \
        --segment-file "../<path_to_segment>.csv" \
        --data-dir "../<you_data_dir>/" \
        --workers 8
```

If the model was trained on a non-DNA alphabet, pass the same `--alphabet`, `--case-sensitive` and
`--complement` values that were used for training. See [Alphabets](#alphabets).


```bash

### R package

To use PGSE through the R package, consult the package
[documentation](https://github.com/yinzheng-zhong/PGSE/tree/main/R-package/).

## For Development

PGSE uses [uv](https://docs.astral.sh/uv/) for packaging and dependency
management. All project metadata and dependencies live in `pyproject.toml`.

Clone the repository and create a synced environment. This installs the runtime
and `dev` dependencies and performs an editable install of `pgse`:
```bash
uv sync
```

The editable install compiles the Aho-Corasick C library
(`pgse/c_lib/aho_corasick.c`) for your platform as part of the install, so the
fast C implementation is used automatically. This requires a C compiler
(`cc`/`gcc`/`clang`; set the `CC` environment variable to choose one). If no
compiler is available the install still succeeds and PGSE transparently falls
back to the slower pure-Python implementation.

To run an editable install on its own:
```bash
uv pip install -e .
```

To build the distribution artifacts (the wheel bundles the compiled library):
```bash
uv build
```

Run the tests with:
```bash
uv run pytest
```

Releases to PyPI are automated by GitHub Actions: when the `version` in
`pyproject.toml` changes on `main`, the workflow compiles the library for all
three platforms, builds the wheel, and publishes it.

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

### Why do we eliminate features?

If segment `A` is extended into segment `B`, `A` becomes a subsequences of `B`. For pairs like `A` and `B`, we only need to keep the
ones with higher feature importance. Extension and elimination are two crucial parts of the PGSE system, which grows the
genome segments longer and the elimination process guarantees that the growth will stop eventually. Additionally, elimination
guarantees the convergence of the system as the feature dimensionality will start decreasing at some point till
all features stop growing.
