import os
from pgse.environment.args import get_parser
import pandas as pd
import json

import math

from pgse.model.util import essential_agreement_cus_metric
from pgse import TrainingPipeline as PGSEPipeline
from pgse import InferencePipeline as PGSEInferencePipeline

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # load label and k-fold info
    label_path = args.label_file
    k_fold_path = args.pre_kfold_info_file

    label_pd = pd.read_csv(label_path)
    k_fold_json = json.load(open(k_fold_path, 'r'))

    inner_folds_file = args.pre_kfold_info_file.replace('.json', '_inner.json')
    export_files = [args.export_file + f'_inner_{i}' for i in range(args.folds)]

    # accumulated results pandas
    results = pd.DataFrame(columns=['Prediction', 'Actual'])

    # k-fold
    for i, fold in enumerate(k_fold_json):
        holdout = k_fold_json[f'fold_{i}']

        # copy the k-fold json
        inner_folds = k_fold_json.copy()
        del inner_folds[f'fold_{i}']

        # get inner fold ids and remove current fold
        inner_folds_ids = list(range(args.folds))
        inner_folds_ids.remove(i)

        # rotate the list so next fold becomes first
        # e.g. at the third fold, the inner folds are [0, 1, 3, 4] -> [3, 4, 0, 1]
        # find the index of the next fold (i+1) or 0 if at end
        next_fold = (i + 1) % args.folds
        if next_fold == i:  # skip if next_fold is the current fold
            next_fold = (i + 2) % args.folds

        rotate_idx = inner_folds_ids.index(next_fold)
        inner_folds_ids = inner_folds_ids[rotate_idx:] + inner_folds_ids[:rotate_idx]

        # reindex the inner folds
        inner_folds = {f'fold_{i}': inner_folds[f'fold_{current}'] for i, current in enumerate(inner_folds_ids)}

        # save the inner fold indices
        with open(inner_folds_file, 'w') as f:
            json.dump(inner_folds, f)

        # create pipeline
        pgse_pipe = PGSEPipeline(
            data_dir=args.data_dir,
            label_file=args.label_file,
            pre_kfold_info_file=inner_folds_file,
            save_file=args.save_file,
            export_file=export_files[i],
            k=args.k,
            ext=args.ext,
            target=args.target,
            features=args.features,
            folds=0,
            ea_min=args.ea_min,
            ea_max=args.ea_max,
            num_rounds=args.num_rounds,
            lr=args.lr,
            dist=False,
            nodes=args.nodes,
            workers=args.workers
        )

        # run pipeline
        pgse_pipe.run()

        holdout_files = [f'{args.data_dir}/{file}' for file in holdout]
        # holdout_labels = label_pd[label_pd['files'].isin(holdout)]['labels'].values
        holdout_labels = []
        for file in holdout:
            label = label_pd[label_pd['files'] == file]['labels'].values
            if len(label) == 0:
                raise ValueError(f'Label not found for file: {file}')
            holdout_labels.append(label[0])

        if len(holdout_files) != len(holdout_labels):
            raise ValueError('Holdout files and labels do not match')

        # inference
        pgse_inference_pipe = PGSEInferencePipeline(
            model_path=export_files[i] + '_fold_0.json',
            segment_path=export_files[i] + '_fold_0.txt',
        )

        # run inference
        out = pgse_inference_pipe.run(holdout_files)

        # add to results using concat instead of append
        new_results = pd.DataFrame({'Prediction': out, 'Actual': holdout_labels})
        results = pd.concat([results, new_results], ignore_index=True)

        # save new_results
        new_results.to_csv(str.replace(export_files[i], 'inner', 'outer') + '.csv', index=False)
        ea = essential_agreement_cus_metric(out, holdout_labels,
                                            min_after_log2=math.log2(args.ea_min) if args.ea_min else None,
                                            max_after_log2=math.log2(args.ea_max) if args.ea_max else None)

        print(f'Fold {i} completed.')

        # remove the save file
        os.remove(args.save_file + ".progress")

    # save results
    results.to_csv(args.export_file + '_nested_test.csv', index=False)

    # concatenate all validation predictions
    val_result = pd.concat([pd.read_csv(export_files[i] + '.csv') for i in range(args.folds)], ignore_index=True)
    val_result.to_csv(args.export_file + '_nested_val.csv', index=False)