# Descriptor Transformer
![CI](https://github.com/buganart/descriptor-transformer/workflows/CI/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/buganart/descriptor-transformer/branch/main/graph/badge.svg)](https://codecov.io/gh/buganart/descriptor-transformer)

Extract audio descriptors and learn to generate them with transformers.

## Development

### Generic

Install `python >= 3.6` and `pytorch` with GPU support if desired.

    pip install -r requirements.txt

Run the tests

    pytest

### Using nix and direnv

1. Install the [nix](https://nixos.org/download.html) package manager
and [direnv](https://direnv.net/).
2. [Hook](https://direnv.net/docs/hook.html) `direnv` into your shell.
3. Type `direnv allow` from within the checkout of this repository.

## Usage

First extract the audio features from `.wav` files in a folder (example
`my_folder`) and save them to a file (example `my_features.npy`).

    python extract_features.py my_folder my_features.npy

To train the model and visualize predictions overlayed with real data

    ./run features.npy

Note: at the moment this concatenates all the features and thus generates some
batches that share data from multiple audio files.

To train the model only, run

    python train.py my_features.npy
