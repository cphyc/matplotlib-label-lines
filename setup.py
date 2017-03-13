# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='matplotlib-label-lines',
    version='0.1.0',
    description='Label lines in matplotlib.',
    author='NauticalMile',
    url='https://github.com/cphyc/matplotlib-label-lines',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy',
        'matplotlib'
    ],
    include_package_data=True
)
