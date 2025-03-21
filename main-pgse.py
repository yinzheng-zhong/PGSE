from pgse.enviromnet import args
from pgse import TrainingPipeline

if __name__ == "__main__":
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
