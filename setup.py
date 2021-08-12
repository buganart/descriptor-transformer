from setuptools import setup, find_packages

setup(
    name="desc",
    version="0.1.0",
    url="https://github.com/buganart/descriptor-transformer",
    author="buganart",
    description="Audio descriptor generating trasnformer",
    packages=find_packages(),
    install_requires=[
        "librosa==0.8.0",
        "matplotlib==3.3.2",
        "pandas==1.1.3",
        "soundfile",
        "torch",
        "tqdm",
        "wandb==0.10.33",
        "pytorch_lightning==1.2.4",
        "numpy",
    ],
    # entry_points={"console_scripts": []},
)
