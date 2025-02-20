# Progressive Genome Segment Enhancement (PGSE)

## Overview
PGSE is a tool for predictive tasks using
whole genome sequences (WGS). Designed and implemented by Y Zhong and originally used for predicting the minimum
inhibitory concentration (MIC) of antibiotics. PGSE has higher accuracy, lower memory consumption and shorter runtime compared to traditional
pure k-mer based xgboost models. PGSE is also able to run on a distributed system.

## Installation
Make sure Python is installed (3.9 or later) and install the required packages using pip:
```bash
pip install -r requirements.txt
```

## Usage
### Training

#### Single node/machine
To run PGSE on a local machine, use the following command as an example:
```bash
python3 main-pege.py \
        --label-file "../<path_to>/<you_labels>.csv" \
        --data-dir "../<you_data_dir>/" \
        --pre-kfold-info-file "../<k_fold_information>.json" \
        --save-file "../<saved progress>.save" \
        --export-file "../<exported files>" \
        --workers $WORKERS_PER_NODE \
        --features 10000 \
        --dist 0 \
        --k 6 \
        --target 70 \
        --ext 2 \
        --lr 0.001 \
        --num-rounds 6000 \
        --folds 5 \
        --ea-max $EA_MAX \
        --ea-min $EA_MIN
```
* `--label-file` (Required): path to the label file

    Here the label file is a csv file with the following format:
    ```csv
    labels, files
    7, file1.fna
    7, file2.fna
    6, file3.fna
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