from setuptools import setup, find_packages

setup(
    name="tidaldub",
    version="0.1.0",
    description="Fully local video dubbing pipeline with enterprise-grade reliability",
    author="TidalDub Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0",
        "sqlalchemy>=2.0",
        "redis>=4.5",
        "click>=8.0",
        "rich>=13.0",
        "flask>=2.3",
        "python-dateutil>=2.8",
        "filelock>=3.12",
    ],
    entry_points={
        "console_scripts": [
            "tidaldub=tidaldub.cli:main",
        ],
    },
)
