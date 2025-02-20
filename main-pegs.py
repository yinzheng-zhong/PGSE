from src.enviromnet import args
from src.pipeline.pegs_pipeline import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline(
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
