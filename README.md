# Progressive Genome Segment Enhancement (PGSE)

## Overview
PGSE is a tool for predictive tasks using
whole genome sequences (WGS). Designed and implemented by Y Zhong and originally used for predicting the minimum
inhibitory concentration (MIC) of antibiotics. PGSE has higher accuracy, lower memory consumption and shorter runtime compared to traditional
pure k-mer based xgboost models. PGSE is also able to run on a distributed system.

## Installation
Make sure Python is installed (3.9 or later) and install the pgse from PyPI:
```bash
pip install pgse
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


Alternatively, to run PGSE as a standalone program on a local machine, download the project and use the following command as an example:
```bash
python3 main-pgse.py \
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
SEGMENT_PATH = '../volatile/var/result-k6-CAZ-perf_fold_0.txt'

if __name__ == "__main__":
    # Instantiate the pipeline
    pipeline = InferencePipeline(MODEL_PATH, SEGMENT_PATH)

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

## For Development
To build the package, run the following command:
```bash
rm -rf dist/ build/ pgse.egg-info/
python -m build
```
Then upload the package to PyPI using:
```bash
python -m twine upload dist/*
```

To install the package locally, run:
```bash
pip install -e .
```

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