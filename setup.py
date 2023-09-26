import os
import sys

from setuptools import setup, find_packages

base_dir = os.path.dirname(__file__)
src_dir = os.path.join(base_dir, 'src')
sys.path.insert(0, src_dir)

import crisis_prediction

SHORT = 'crisis_prediction'

authors = [
    'Koa Health'
]


__author__ = ', '.join(authors)


def get_requirements(requirements_path: str = 'requirements.txt'):
    with open(requirements_path) as fp:
        return [x.strip() for x in fp.read().split('\n') if not x.startswith('#')]


setup(
    name='crisis_prediction',
    version=crisis_prediction.__version__,
    packages=find_packages(where='src'),
    install_requires=get_requirements(),
    url='https://github.com/Icedgarr/crisis_prediction.git',
    package_dir={'': 'src'},
    description=SHORT,
    long_description="Code used to generate the results of the article 'Combining clinical notes with structured electronic health records enhances the prediction of mental health crises'",
)
