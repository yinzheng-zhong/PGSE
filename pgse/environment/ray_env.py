import os
import ray
from pgse.log import logger


class RayEnvManager:
    @staticmethod
    def initialize(dist: bool, nodes: int, workers: int):
        # skip if already initialized
        if ray.is_initialized():
            return
        os.environ["RAY_LOG_TO_STDERR"] = "0"
        os.environ["RAY_LOG_LEVEL"] = "ERROR"

        if dist:
            ray.init(address='auto', log_to_driver=True)
            logger.warning(
                f'Connected to Ray cluster with {nodes} nodes and {workers} workers per node.\n'
                f'Sometimes the progress bar may seem frozen, but it is still running.'
            )
        else:
            ray.init(num_cpus=workers, log_to_driver=True)
