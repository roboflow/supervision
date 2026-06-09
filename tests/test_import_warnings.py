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
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import supervision  # noqa: F401

        offenders = [
            str(w.message)
            for w in caught
            if w.category is FutureWarning and "TargetMode" in str(w.message)
        ]
        if offenders:
            raise SystemExit(
                "legacy target= sentinel warnings on import:\\n" + "\\n".join(offenders)
            )
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter + trusted literal code
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
