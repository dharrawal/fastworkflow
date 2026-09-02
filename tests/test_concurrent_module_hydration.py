"""Two threads hydrating sibling command modules must not see half-built work.

Bead ido-2l0.28, found by the first G2B generation losing 12 of 16 attempts.

`python_utils.get_module` loads a module by calling `spec.loader.exec_module`
directly. That is a legitimate thing to do — the workflow's command modules live
outside any importable package root — but it bypasses the per-module lock
importlib holds around a normal import, and the module is published to
`sys.modules` *before* it is executed so that relative imports inside it
resolve.

Those two facts together are the defect. Thread A publishes an empty module and
starts executing it; thread B imports a sibling whose `from .that_module import
Thing` finds the empty module already in `sys.modules`, takes it for finished,
and fails with `cannot import name 'Thing'` — naming something that is plainly
in the file.

Measured on the IDO tree before the fix: 3 to 7 failures per 8 concurrent cold
loads. After: zero, repeatedly.
"""

from __future__ import annotations

import sys
import textwrap
import threading

import pytest

from fastworkflow.utils.python_utils import get_module


@pytest.fixture
def sibling_modules(tmp_path):
    """A package whose second module imports a name from its first.

    Shaped like the real case: `assess_risk` does
    `from .list_controls_hit import ResponseGenerator`.
    """
    pkg = tmp_path / "wf" / "_commands"
    pkg.mkdir(parents=True)
    (tmp_path / "wf" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    # A body slow enough that a second thread reliably lands inside the window
    # between "published to sys.modules" and "finished executing".
    (pkg / "provider.py").write_text(
        textwrap.dedent(
            """
            import time
            time.sleep(0.25)


            class ResponseGenerator:
                pass
            """
        )
    )
    (pkg / "composite.py").write_text(
        textwrap.dedent(
            """
            from .provider import ResponseGenerator as Provided


            class ResponseGenerator:
                inner = Provided
            """
        )
    )
    yield tmp_path
    for name in list(sys.modules):
        if name.startswith("wf."):
            del sys.modules[name]


def test_a_sibling_import_never_sees_a_half_built_module(sibling_modules):
    """The race, driven directly.

    One thread loads the slow provider; the other loads the composite that
    imports a name out of it. Without the per-module lock the second thread
    gets the empty module and raises.
    """
    root = str(sibling_modules)
    errors: list[str] = []
    barrier = threading.Barrier(2)

    def load(relative: str):
        try:
            barrier.wait()
            get_module(f"{root}/wf/_commands/{relative}", root)
        except BaseException as exc:  # noqa: BLE001 - the assertion is the point
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [
        threading.Thread(target=load, args=("provider.py",)),
        threading.Thread(target=load, args=("composite.py",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == [], errors


def test_many_concurrent_loads_of_the_same_module_agree(sibling_modules):
    """Eight threads, one module: they must all get the same finished object.

    Not merely "no exception" — two threads each executing their own copy would
    also raise nothing, and would leave two different classes with the same
    name in one process.
    """
    root = str(sibling_modules)
    seen: list[object] = []
    errors: list[str] = []
    barrier = threading.Barrier(8)

    def load():
        try:
            barrier.wait()
            seen.append(get_module(f"{root}/wf/_commands/provider.py", root))
        except BaseException as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=load) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == [], errors
    assert len(seen) == 8
    assert all(module is seen[0] for module in seen)


def test_a_failed_import_is_not_left_in_sys_modules(tmp_path):
    """A half-built module left behind poisons every later import of it.

    This is what turned one racing failure into a whole generation of them: the
    first thread's broken module stayed cached, so every attempt afterwards in
    that process failed the same way, long after the race itself was over.
    """
    pkg = tmp_path / "wf" / "_commands"
    pkg.mkdir(parents=True)
    (tmp_path / "wf" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    broken = pkg / "broken.py"
    broken.write_text("raise RuntimeError('boom')\n")

    with pytest.raises(ImportError):
        get_module(f"{tmp_path}/wf/_commands/broken.py", str(tmp_path))

    assert "wf._commands.broken" not in sys.modules

    # And the name is reusable: a repaired file imports cleanly rather than
    # being permanently shadowed by the failure.
    broken.write_text("VALUE = 1\n")
    get_module.cache_clear()
    module = get_module(f"{tmp_path}/wf/_commands/broken.py", str(tmp_path))
    assert module.VALUE == 1

    for name in list(sys.modules):
        if name.startswith("wf."):
            del sys.modules[name]
