# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open("Readme.md") as file:
    long_description = file.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="matplotlib-label-lines",
    version="0.3.9",
    description="Label lines in matplotlib.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Corentin Cadiou",
    author_email="contact@cphyc.me",
    url="https://github.com/cphyc/matplotlib-label-lines",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=["numpy", "matplotlib"],
    include_package_data=True,
)
