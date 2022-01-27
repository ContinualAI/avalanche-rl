import setuptools
import codecs
import os.path

_VERSION = "0.0.1"

with open("README.md", "r") as fh:
    long_description = fh.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="avalanche-rl", 
    version=_VERSION,
    author="ContinualAI",
    author_email="contact@continualai.org",
    description="Avalanche RL: an End-to-End Library for Continual Reinforcement Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ContinualAI/avalanche-rl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # python_requires='>=3.6,<=3.9.2',
    install_requires=[
        'avalanche-lib',
        'gym',
        'ray',
    ]
)
