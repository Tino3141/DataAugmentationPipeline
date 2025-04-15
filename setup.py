from setuptools import setup, find_packages

setup(
    name="data_augmentation_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        "numpy",
        "pydub",
    ],
) 