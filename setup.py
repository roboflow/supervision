import setuptools
from setuptools import find_packages
import re

with open('./supervision/__init__.py', 'r') as f:
    content = f.read()
    # from https://www.py4u.net/discuss/139845
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)
    
with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='supervision',
    version=version,
    author='Piotr Skalski',
    author_email='piotr.skalski92@gmail.com',
    license='MIT',
    description='A set of easy-to-use utils that will come in handy in any Computer Vision project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/roboflow-ai/supervision',
    install_requires=[],
    packages=find_packages(exclude=("tests",)),
    extras_require={
        'annotators': [
            'numpy',
            'opencv-python'
        ],
        'dev': [
            'flake8',
            'black==22.3.0',
            'isort',
            'twine',
            'pytest',
            'wheel',
            'notebook'
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Typing :: Typed',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    python_requires='>=3.7',
)
