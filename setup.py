"""
## Build Lifecycle

pip install . / uv pip install .
        ↓
setup.py cmdclass hooks triggered
        ↓
CustomBuildPy.run() or CustomSdist.run()
        ↓
_pre_build()  ← YOUR CODE RUNS HERE
        ↓
Files packaged into wheel/sdist
"""
import re
from pathlib import Path
from setuptools import setup
from setuptools.command.sdist import sdist
from setuptools.command.build_py import build_py


class PreBuildMixin:
    """Shared pre-build logic for both sdist and wheel builds."""

    def _pre_build(self):
        print("🔧 Running pre-build modifications...")
        self._update_readme()
        self._update_requirements()
        print("✅ Pre-build complete!")

    def _update_readme(self):
        readme = Path("README.md")
        if readme.exists():
            content = readme.read_text()
            # Example: update branch links to tag
            updated = re.sub(
                r"(https://github\.com/user/repo/tree/)main",
                r"\1v0.1.0",  # or dynamically get version
                content
            )
            readme.write_text(updated)
            print("  📝 Updated README links")

    def _update_requirements(self):
        req_file = Path("requirements.txt")
        if req_file.exists():
            content = req_file.read_text()
            # Example: pin or adjust versions
            updated = content.replace("torch>=2.0", "torch>=2.1,<3.0")
            req_file.write_text(updated)
            print("  📦 Updated requirements")


class CustomSdist(PreBuildMixin, sdist):
    """Hook for source distribution (pip install from sdist)."""

    def run(self):
        self._pre_build()
        super().run()


class CustomBuildPy(PreBuildMixin, build_py):
    """Hook for wheel build."""

    def run(self):
        self._pre_build()
        super().run()


setup(
    cmdclass={
        "sdist": CustomSdist,
        "build_py": CustomBuildPy,
    }
)