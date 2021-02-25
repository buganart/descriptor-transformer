from setuptools import setup, find_packages

setup(
    name="desc",
    version="0.1.0",
    url="https://github.com/buganart/descriptor-transformer",
    author="buganart",
    description="Audio descriptor generating trasnformer",
    packages=find_packages(),
    install_requires=[
        "librosa",
        "matplotlib",
        "pandas",
        "soundfile",
        "torch",
        "tqdm",
        "wandb",
        "pytorch_lightning",
        "numpy",
    ],
    # entry_points={"console_scripts": []},
)
