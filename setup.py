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
    package_data={"minder_utils.configurations": ["*.yaml"]},
    long_description=open('README.txt').read(),
    install_requires=["requests",
                        "argparse",
                        "pyyaml",
                        "typing-extensions==3.7.4",
                        "pydtmc",
                        "networkx==2.6.3"
    ]
)
