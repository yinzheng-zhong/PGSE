import os
import ray
from src.enviromnet import args
from src.log import logger


class RayEnvManager:
    @staticmethod
    def initialize():
        os.environ["RAY_LOG_TO_STDERR"] = "0"
        os.environ["RAY_LOG_LEVEL"] = "ERROR"

        if args.dist:
            ray.init(address='auto', log_to_driver=True)
            logger.warning(
                f'Connected to Ray cluster with {args.nodes} nodes and {args.workers} workers per node.\n'
                f'Sometimes the progress bar may seem frozen, but it is still running.'
            )
        else:
            ray.init(num_cpus=args.workers, log_to_driver=True)
