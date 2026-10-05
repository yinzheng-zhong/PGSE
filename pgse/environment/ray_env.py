import os
from typing import Optional

import ray

from pgse.environment.ray_memory import GIB, object_store_memory
from pgse.log import logger


class RayEnvManager:
    # Whether PGSE, rather than the host application, called ray.init().
    _started_by_pgse: bool = False

    # Fill fraction of the object store at which Ray starts spilling objects to disk.
    SPILL_THRESHOLD: float = 0.95

    @staticmethod
    def initialize(dist: bool, nodes: int, workers: int, spill_dir: Optional[str] = None) -> None:
        """Start a local Ray instance, or connect to a running cluster.

        Does nothing when Ray is already running. RAY_object_spilling_threshold in the
        environment overrides SPILL_THRESHOLD.

        Args:
            dist: Connect to the running multi-node cluster instead of starting Ray here.
            nodes: Number of nodes in the cluster, for the log.
            workers: Number of CPUs the local Ray instance may use.
            spill_dir: Directory the local Ray instance spills objects to. None keeps Ray's
                default, its session directory under the system temp directory. Not used
                with dist, where the cluster's own setting applies.
        """
        # skip if already initialized
        if ray.is_initialized():
            return
        os.environ["RAY_LOG_TO_STDERR"] = "0"
        os.environ["RAY_LOG_LEVEL"] = "ERROR"
        os.environ.setdefault("RAY_object_spilling_threshold", str(RayEnvManager.SPILL_THRESHOLD))

        if dist:
            ray.init(address='auto', log_to_driver=True)
            logger.warning(
                f'Connected to Ray cluster with {nodes} nodes and {workers} workers per node.\n'
                f'Sometimes the progress bar may seem frozen, but it is still running.'
            )
        else:
            store_memory = object_store_memory()
            if store_memory:
                logger.info(f'Ray object store: {store_memory / GIB:.0f} GiB')

            spill_path = None
            if spill_dir:
                spill_path = os.path.abspath(spill_dir)
                os.makedirs(spill_path, exist_ok=True)
                logger.info(f'Ray spills objects to {spill_path}')

            ray.init(
                num_cpus=workers,
                object_store_memory=store_memory,
                object_spilling_directory=spill_path,
                log_to_driver=True
            )

        RayEnvManager._started_by_pgse = True

    @staticmethod
    def shutdown() -> None:
        """Shut Ray down, unless the host application was the one that started it."""
        if RayEnvManager._started_by_pgse and ray.is_initialized():
            ray.shutdown()

        RayEnvManager._started_by_pgse = False
