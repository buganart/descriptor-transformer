from setuptools import setup, find_packages

setup(
    name="dialog",
    version="0.1.0",
    url="https://github.com/buganart/descriptor-transformer",
    author="buganart",
    description="Audio descriptor generating trasnformer",
    packages=find_packages(),
    install_requires=[
        "librosa",
        "soundfile",
        "torch",
        "tqdm",
        "wandb",
    ],
    # entry_points={"console_scripts": []},
)
