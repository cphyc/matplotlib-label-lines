# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('Readme.md') as f:
    readme = f.read()


setup(
    name='matplotlib-label-lines',
    version='0.2.0',
    description='Label lines in matplotlib.',
    long_description=readme,
    author='Corentin Cadiou',
    url='https://github.com/cphyc/matplotlib-label-lines',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy',
        'matplotlib'
    ],
    include_package_data=True
)
