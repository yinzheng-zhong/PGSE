"""Sizing of the Ray object store from the memory available to this process."""

import os
import shutil
import sys
from typing import Optional

from pgse.log import logger

OBJECT_STORE_FRACTION = 0.4
SHM_SAFETY_FRACTION = 0.95
GIB = 1024 ** 3

PROC_CGROUP = '/proc/self/cgroup'
CGROUP_ROOT = '/sys/fs/cgroup'


def physical_memory() -> Optional[int]:
    """Total physical memory in bytes, or None when it cannot be determined."""
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (AttributeError, ValueError, OSError):
        pass

    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def _limits_up_tree(directory: str, file_name: str, top: str) -> list[int]:
    """Numeric limits read from file_name in directory and each of its parents up to top.

    Args:
        directory: The cgroup directory to start from.
        file_name: The limit file, e.g. 'memory.max'.
        top: The directory to stop at, inclusive.
    """
    limits = []
    top = os.path.normpath(top)
    directory = os.path.normpath(directory)
    while True:
        try:
            with open(os.path.join(directory, file_name)) as handle:
                value = handle.read().strip()
            if value.isdigit():
                limits.append(int(value))
        except OSError:
            pass

        parent = os.path.dirname(directory)
        if directory == top or parent == directory:
            return limits
        directory = parent


def cgroup_memory_limit(proc_cgroup: str = PROC_CGROUP, cgroup_root: str = CGROUP_ROOT) -> Optional[int]:
    """Smallest memory limit set on this process's cgroup or any of its parents.

    Reads memory.max under cgroup v2 and memory.limit_in_bytes under cgroup v1.

    Args:
        proc_cgroup: File listing the cgroups of the process.
        cgroup_root: Mount point of the cgroup filesystem.

    Returns:
        The limit in bytes, or None when no limit is set or the files cannot be read.
    """
    try:
        with open(proc_cgroup) as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None

    limits = []
    for line in lines:
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == '':
            top = cgroup_root
            limits += _limits_up_tree(os.path.join(top, path.lstrip('/')), 'memory.max', top)
        elif 'memory' in controllers.split(','):
            top = os.path.join(cgroup_root, 'memory')
            limits += _limits_up_tree(os.path.join(top, path.lstrip('/')), 'memory.limit_in_bytes', top)

    return min(limits) if limits else None


def memory_limit() -> Optional[int]:
    """Memory available to this process in bytes: physical memory or the cgroup limit,
    whichever is smaller. None when neither can be determined."""
    known = [value for value in (physical_memory(), cgroup_memory_limit()) if value]
    return min(known) if known else None


def object_store_memory() -> Optional[int]:
    """Size of the Ray object store in bytes, or None to let Ray decide.

    PGSE_OBJECT_STORE_MEMORY overrides the size. Otherwise, on Linux, it is
    OBJECT_STORE_FRACTION of the memory available to this process, capped at
    SHM_SAFETY_FRACTION of the free space in /dev/shm.
    """
    override = os.environ.get('PGSE_OBJECT_STORE_MEMORY')
    if override:
        try:
            return int(override)
        except ValueError:
            logger.warning(f'Ignoring invalid PGSE_OBJECT_STORE_MEMORY={override!r}.')

    if not sys.platform.startswith('linux'):
        return None

    available = memory_limit()
    if not available:
        logger.warning('Could not determine the available memory. Using Ray defaults.')
        return None

    target = int(available * OBJECT_STORE_FRACTION)
    logger.info(
        f'Ray object store: {OBJECT_STORE_FRACTION:.0%} of the {available / GIB:.0f} GiB '
        f'of memory available to this process.'
    )

    if os.path.isdir('/dev/shm'):
        shm_free = shutil.disk_usage('/dev/shm').free
        cap = int(shm_free * SHM_SAFETY_FRACTION)
        if target > cap:
            target = cap
            logger.warning(
                f'/dev/shm has only {shm_free / GIB:.0f} GiB free, so the Ray object store is '
                f'limited to {target / GIB:.0f} GiB.'
            )

    return target
