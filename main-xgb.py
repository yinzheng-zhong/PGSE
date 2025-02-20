from pgse.enviromnet import args
from pgse.pipeline.regular_pipline import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline(
        args.data_dir,
        args.label_file,
        args.pre_kfold_info_file,
        args.export_file,
        args.k,
        args.ext,
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
