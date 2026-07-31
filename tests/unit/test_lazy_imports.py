import importlib
import subprocess
import sys
import unittest

import pgse

TRAINING_STACK = ('xgboost', 'ray', 'sklearn')


def stack_is_usable() -> bool:
    """True when every training-stack module imports without raising."""
    try:
        for name in TRAINING_STACK:
            importlib.import_module(name)
    except Exception:
        return False
    return True


STACK_USABLE = stack_is_usable()


class TestLazyPipelineImports(unittest.TestCase):
    """
    The counting path must import with only NumPy and SciPy present: neither
    ``import pgse`` nor ``pgse.algos`` may pull in the training stack. The pipelines
    stay reachable as attributes of ``pgse``.
    """

    def loaded_after(self, snippet: str) -> str:
        """The training-stack modules present in ``sys.modules`` after running ``snippet``."""
        probe = (
            f'import sys; {snippet}; '
            f'print(",".join(sorted({set(TRAINING_STACK)!r} & set(sys.modules))))'
        )
        result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_importing_pgse_does_not_import_the_training_stack(self):
        self.assertEqual('', self.loaded_after('import pgse'))

    def test_importing_the_counting_path_does_not_import_the_training_stack(self):
        self.assertEqual('', self.loaded_after('from pgse.algos import native_counter'))

    def test_unknown_attributes_still_raise(self):
        with self.assertRaises(AttributeError):
            pgse.NotAPipeline

    @unittest.skipUnless(STACK_USABLE, 'the training stack cannot be imported here')
    def test_pipelines_are_still_exported(self):
        for name in pgse.__all__:
            with self.subTest(pipeline=name):
                self.assertTrue(callable(getattr(pgse, name)))


if __name__ == '__main__':
    unittest.main()
