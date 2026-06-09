from __future__ import annotations

import subprocess
import sys
import textwrap


def test_import_supervision_emits_no_targetmode_future_warning() -> None:
    """Importing supervision must not emit `deprecate` legacy-``target=`` sentinel
    FutureWarnings.

    ``pydeprecate>=0.9`` (``import deprecate``) deprecated the ``target=True`` /
    ``target=None`` sentinel form of its ``@deprecated`` / ``@deprecated_class``
    decorators in favour of the ``TargetMode`` enum. Those sentinels fire a
    ``FutureWarning`` at decoration time — i.e. on ``import supervision`` for the
    affected modules — and become a hard error in ``pydeprecate`` v1.0. This guards
    against the sentinel form creeping back in. A fresh subprocess is used because
    the module is already imported in-process.
    """
    code = textwrap.dedent(
        """
        import os
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import supervision  # noqa: F401

        sv_path = os.path.dirname(os.path.abspath(supervision.__file__))
        offenders = [
            str(w.message)
            for w in caught
            if w.category is FutureWarning
            and os.path.abspath(w.filename).startswith(sv_path)
        ]
        if offenders:
            detail = "\\n".join(offenders)
            raise SystemExit(f"FutureWarning(s) on import:\\n{detail}")
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter + trusted literal code
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
