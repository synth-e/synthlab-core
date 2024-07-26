# Copyright 2023 parkminwoo, MIT License
from setuptools import find_packages
from setuptools import setup

# read synthlab_core/__version__.py

version = {}

with open("synthlab_core/version.py") as fp:
    exec(fp.read(), version)
    
print(version.keys())
    
def get_long_description():
    with open("README.md", encoding="UTF-8") as f:
        long_description = f.read()
        return long_description

dependencies = [
    "torch",
    "torchvision",
    "nltk",
    "structlog",
    "opencv-python",
    "requests"
]  

visualization_pack = ["grandalf", "pygraphviz"]

setup(
    name="synthlab_core",
    version=version["__version__"],
    author="Ngoc-Do Tran",
    author_email="dotrann.1412@gmail.com",
    description="Somthing extremely cool.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/synth-e/synthlab",
    packages=find_packages(exclude=[], include=["synthlab_core", "synthlab_core.*"]),

    python_requires=">=3.9",
    install_requires=dependencies,
    keywords="Python, API, Computer vision, Modular design",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
)
