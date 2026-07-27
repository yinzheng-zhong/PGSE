"""In-tree PEP 517 build backend for PGSE.

This is a thin wrapper around setuptools' own backend. The only thing it adds is
compiling the Aho-Corasick shared library (``pgse/c_lib/aho_corasick.c``) for the
host platform before a wheel or an editable install is built, so a plain
``uv pip install -e .`` (or ``uv build``) ends up with the fast C implementation
instead of silently falling back to the slower pure-Python one.

Compilation is best effort: PGSE ships a pure-Python fallback, so a missing or
broken compiler produces a loud warning rather than a failed install.

Referenced from ``pyproject.toml`` via::

    [build-system]
    build-backend = "_build_backend"
    backend-path = ["."]
"""

from __future__ import annotations

import os
import subprocess
import sys

# Re-export every hook from setuptools' backend, then override the two that need
# to compile the C library first. Anything not overridden (sdist, metadata, the
# get_requires_* hooks) keeps setuptools' default behaviour.
from setuptools import build_meta as _setuptools_backend
from setuptools.build_meta import (  # noqa: F401  (re-exported for the frontend)
    build_sdist,
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_C_LIB_DIR = os.path.join(_HERE, "pgse", "c_lib")
_C_SOURCE = os.path.join(_C_LIB_DIR, "aho_corasick.c")


def _library_filename() -> str:
    """The ctypes loader (``pgse/algos/aho_corasick_c.py``) looks for these names."""
    if sys.platform.startswith("win"):
        return "aho_corasick.dll"
    if sys.platform.startswith("darwin"):
        return "aho_corasick.dylib"
    return "aho_corasick.so"


def _compile_command(output: str) -> list[str]:
    # Mirror the flags used by the release workflow (.github/workflows). The host
    # compiler is honoured via $CC so cross builds / distro toolchains still work.
    compiler = os.environ.get("CC") or ("gcc" if sys.platform.startswith("win") else "cc")
    flags = ["-shared", "-O3"]
    if not sys.platform.startswith("win"):
        flags.append("-fPIC")
    return [compiler, *flags, "-o", output, _C_SOURCE]


def _compile_c_library() -> None:
    """Compile the shared library in place, warning (not failing) on any error."""
    if not os.path.isfile(_C_SOURCE):
        print(f"[pgse] C source not found at {_C_SOURCE}; skipping compile.", file=sys.stderr)
        return

    output = os.path.join(_C_LIB_DIR, _library_filename())
    command = _compile_command(output)
    print(f"[pgse] Compiling Aho-Corasick C library: {' '.join(command)}", file=sys.stderr)

    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(
            f"[pgse] WARNING: could not build the C library ({error}). PGSE will fall back "
            "to the slower pure-Python implementation. Install a C compiler (gcc/clang) "
            "and reinstall to enable the fast path.",
            file=sys.stderr,
        )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _compile_c_library()
    return _setuptools_backend.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _compile_c_library()
    return _setuptools_backend.build_editable(wheel_directory, config_settings, metadata_directory)
