import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from pgse.environment import ray_memory
from pgse.environment.ray_memory import GIB, cgroup_memory_limit, object_store_memory


def write(path: str, text: str) -> None:
    """Write text to path, creating its parent directories.

    Args:
        path: File to write.
        text: Contents of the file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write(text)


class TestCgroupMemoryLimit(unittest.TestCase):
    """The smallest memory limit found walking up the process's cgroup tree."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, 'cgroup')
        self.proc = os.path.join(self.tmp.name, 'proc_cgroup')

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_v2_reads_the_job_limit_above_an_unlimited_task(self) -> None:
        job = os.path.join(self.root, 'system.slice/slurmstepd.scope/job_1')
        write(self.proc, '0::/system.slice/slurmstepd.scope/job_1/step_0/user/task_0\n')
        write(os.path.join(job, 'memory.max'), '1073741824\n')
        write(os.path.join(job, 'step_0/memory.max'), 'max\n')
        write(os.path.join(job, 'step_0/user/memory.max'), '2147483648\n')
        write(os.path.join(job, 'step_0/user/task_0/memory.max'), 'max\n')

        self.assertEqual(cgroup_memory_limit(self.proc, self.root), 1073741824)

    def test_v1_reads_the_memory_controller(self) -> None:
        job = os.path.join(self.root, 'memory/slurm/uid_1/job_2')
        write(self.proc, '3:cpu,cpuacct:/slurm/uid_1/job_2\n12:memory:/slurm/uid_1/job_2/step_batch\n')
        write(os.path.join(job, 'memory.limit_in_bytes'), '4294967296\n')
        write(os.path.join(job, 'step_batch/memory.limit_in_bytes'), '9223372036854771712\n')

        self.assertEqual(cgroup_memory_limit(self.proc, self.root), 4294967296)

    def test_no_limit_is_none(self) -> None:
        write(self.proc, '0::/user.slice/session-1.scope\n')
        write(os.path.join(self.root, 'user.slice/memory.max'), 'max\n')
        write(os.path.join(self.root, 'user.slice/session-1.scope/memory.max'), 'max\n')

        self.assertIsNone(cgroup_memory_limit(self.proc, self.root))

    def test_unreadable_proc_file_is_none(self) -> None:
        self.assertIsNone(cgroup_memory_limit(self.proc, self.root))


class TestObjectStoreMemory(unittest.TestCase):
    """The object store size: a fraction of the available memory, capped by free /dev/shm."""

    def size_with(self, available: int, shm_free: int, platform: str = 'linux') -> object:
        """object_store_memory() with the available memory and /dev/shm free space faked.

        Args:
            available: Bytes of memory available to the process.
            shm_free: Bytes free in /dev/shm.
            platform: Value of sys.platform.
        """
        with mock.patch.dict(os.environ), \
                mock.patch.object(ray_memory.sys, 'platform', platform), \
                mock.patch.object(ray_memory, 'memory_limit', return_value=available), \
                mock.patch.object(ray_memory.os.path, 'isdir', return_value=True), \
                mock.patch.object(ray_memory.shutil, 'disk_usage', return_value=SimpleNamespace(free=shm_free)):
            os.environ.pop('PGSE_OBJECT_STORE_MEMORY', None)
            return object_store_memory()

    def test_fraction_of_available_memory(self) -> None:
        self.assertEqual(self.size_with(100 * GIB, 1000 * GIB), int(100 * GIB * ray_memory.OBJECT_STORE_FRACTION))

    def test_capped_by_free_shared_memory(self) -> None:
        self.assertEqual(self.size_with(1000 * GIB, 50 * GIB), int(50 * GIB * ray_memory.SHM_SAFETY_FRACTION))

    def test_left_to_ray_off_linux(self) -> None:
        self.assertIsNone(self.size_with(100 * GIB, 1000 * GIB, platform='darwin'))

    def test_environment_override(self) -> None:
        with mock.patch.dict(os.environ, {'PGSE_OBJECT_STORE_MEMORY': '12345'}):
            self.assertEqual(object_store_memory(), 12345)


if __name__ == '__main__':
    unittest.main()
