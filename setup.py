from setuptools import setup, find_packages

setup(
    name="ruzzianprop",
    version="0.1.0",
    description="Russian disinformation bot network analyser",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "tweepy>=4.14.0",
        "networkx>=3.1",
        "matplotlib>=3.7.0",
        "pyvis>=0.3.2",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "python-louvain>=0.16",
        "click>=8.1.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "python-dateutil>=2.8.2",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "ruzzianprop=src.cli:cli",
        ],
    },
)
