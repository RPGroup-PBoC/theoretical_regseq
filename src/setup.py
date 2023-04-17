import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tregs",
    version="0.0.1",
    author="Rosalind Pan, Tom Roeschinger",
    author_email="rosalindpan@caltech.edu",
    description="This repository contains the code for the theoretical Reg-Seq project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RPGroup-PBoC/theoretical_regseq",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)