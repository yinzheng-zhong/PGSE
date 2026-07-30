import os
import shutil
import sys
from typing import Optional

import ray
from pgse.log import logger

# Ray sizes its object store at min(30% of RAM, 200GB) by default. On a large-memory
# node the 200GB cap leaves most of the machine idle and pushes objects to disk, so
# claim a larger share instead.
OBJECT_STORE_FRACTION = 0.4

# Ray's object store is backed by /dev/shm on Linux. Asking for more than /dev/shm can
# hold makes Ray fall back to a disk-backed file, which is far slower than spilling.
SHM_SAFETY_FRACTION = 0.95

GIB = 1024 ** 3


def _get_total_memory() -> Optional[int]:
    """
    Total physical memory in bytes, without pulling in a dependency for it.

    :return: int or None: The total memory, or None if it could not be determined.
    """
    try:
        # POSIX (Linux and macOS). Not available on Windows.
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (AttributeError, ValueError, OSError):
        pass

    try:
        import psutil  # Not a declared Ray dependency, so only a fallback.
        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def _get_object_store_memory() -> Optional[int]:
    """
    Size the object store from the machine rather than Ray's 200GB default cap.

    :return: int or None: The object store size in bytes, or None to let Ray decide.
    """
    override = os.environ.get('PGSE_OBJECT_STORE_MEMORY')
    if override:
        try:
            return int(override)
        except ValueError:
            logger.warning(f'Ignoring invalid PGSE_OBJECT_STORE_MEMORY={override!r}.')

    if not sys.platform.startswith('linux'):
        # Ray refuses a configured object store above 2GiB on macOS, and the large
        # shared-memory store this sizing targets is a Linux/HPC concern anyway.
        return None

    total = _get_total_memory()
    if not total:
        logger.warning('Could not determine the total memory. Using Ray defaults.')
        return None

    target = int(total * OBJECT_STORE_FRACTION)

    # Cap by /dev/shm, otherwise Ray silently switches to a disk-backed store.
    if os.path.isdir('/dev/shm'):
        shm_total = shutil.disk_usage('/dev/shm').total
        if target > shm_total * SHM_SAFETY_FRACTION:
            target = int(shm_total * SHM_SAFETY_FRACTION)
            logger.warning(
                f'/dev/shm holds only {shm_total / GIB:.0f} GiB, so the Ray object store is '
                f'limited to {target / GIB:.0f} GiB. Increase /dev/shm (e.g. --shm-size for '
                f'containers) to keep more objects in memory.'
            )

    return target


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
            object_store_memory = _get_object_store_memory()
            if object_store_memory:
                logger.info(f'Ray object store: {object_store_memory / GIB:.0f} GiB')
            ray.init(
                num_cpus=workers,
                object_store_memory=object_store_memory,
                log_to_driver=True
            )
