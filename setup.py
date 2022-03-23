# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kilt",
    version="0.1.0",
    description="Knowledge Intensive Language Tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "bs4",
        "flair",
        "jsonlines",
        "nltk",
        "prettytable",
        "pymongo",
        "pytest",
        "rouge",
        "spacy>=2.1.8",
        "torch",
        "tqdm",
    ],
)
