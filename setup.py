import setuptools
from setuptools import find_packages
import re
from pathlib import Path

FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / "README.md").read_text(encoding="utf-8")


def get_version():
    file = PARENT / 'supervision/__init__.py'
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding="utf-8"), re.M)[1]
    

setuptools.setup(
    name='supervision',
    version=get_version(),
    author='Piotr Skalski',
    author_email='piotr.skalski92@gmail.com',
    license='MIT',
    description='A set of easy-to-use utils that will come in handy in any Computer Vision project',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/roboflow/supervision',
    install_requires=[
        'numpy>=1.20.0',
        'opencv-python',
        'matplotlib'
    ],
    packages=find_packages(exclude=("tests",)),
    extras_require={
        'dev': [
            'flake8',
            'black==22.3.0',
            'isort',
            'twine',
            'pytest',
            'wheel',
            'notebook',
            'mkdocs-material',
            'mkdocstrings[python]'
        ],
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        'Typing :: Typed',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    keywords="machine-learning, deep-learning, vision, ML, DL, AI, YOLOv5, YOLOv8, Roboflow",
    python_requires='>=3.7',
)
