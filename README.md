# Descriptor Transformer
![CI](https://github.com/buganart/descriptor-transformer/workflows/CI/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/buganart/descriptor-transformer/branch/master/graph/badge.svg)](https://codecov.io/gh/buganart/descriptor-transformer)

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
