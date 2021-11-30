import os
from setuptools import setup, find_packages
import subprocess
import logging

PACKAGE_NAME = 'minder_utils'


setup(
    name=PACKAGE_NAME,
    version='0.0.2',
    description='A package for loading the DRI data from Minder',
    author='UKDRI CR&T Imperial College London',
    author_email='',
    url='https://github.com/ImperialCollegeLondon/minder_utils',
    packages=['minder_utils',],
    long_description=open('README.txt').read(),
    install_requires=["numpy==1.21.4",
                        "pandas==1.1.5",
                        "requests",
                        "argparse",
                        "pyyaml"
    ]
)