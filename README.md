# Descriptor Transformer
![CI](https://github.com/buganart/descriptor-transformer/workflows/CI/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/buganart/descriptor-transformer/branch/main/graph/badge.svg)](https://codecov.io/gh/buganart/descriptor-transformer)

Extract audio descriptors and learn to generate them with transformers.

## Development

### Installation

    pip install -e .


### Install package and development tools

Install `python >= 3.6` and `pytorch` with GPU support if desired.

    pip install -r requirements.txt


<!-- Run the tests

    pytest -->

<!-- 
### Option 2: Using nix and direnv

1. Install the [nix](https://nixos.org/download.html) package manager
and [direnv](https://direnv.net/).
2. [Hook](https://direnv.net/docs/hook.html) `direnv` into your shell.
3. Type `direnv allow` from within the checkout of this repository. -->


## Usage

From the descripton above, descriptor model is necessary for the prediction workflow. User can use one of the pretrained descriptor model with the wandb run id in the [prediction notebook](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/descriptor_model_predict.ipynb), or train their own model with the instruction in the training section below.

For the descriptor model, there are 4 models to choose from: "LSTM", "LSTMEncoderDecoderModel", "TransformerEncoderOnlyModel", or "TransformerModel".
The "LSTM" and "TransformerEncoderOnlyModel" are one step prediction model, while "LSTMEncoderDecoderModel" and "TransformerModel" can predict descriptor sequence with specified sequence length.

After training the model, record the wandb run id and paste it in the [prediction notebook](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/descriptor_model_predict.ipynb). Then, provide paths to the RAW generated audio DB and Prediction DB, and run the notebook. The notebook will generate new descriptors from the descriptor model and convert them back into audio.

### Training (notebook)

The [training notebook](https://github.com/buganart/descriptor-transformer/blob/main/train_notebook/descriptor_model_train.ipynb) for the descriptor model is located in the folder [train_notebook/](https://github.com/buganart/descriptor-transformer/tree/main/train_notebook).

Follow the instruction in the [training notebook](https://github.com/buganart/descriptor-transformer/blob/main/train_notebook/descriptor_model_train.ipynb) to train the descriptor model.

### Training (script)

To train the descriptor model, run

    python desc/train_function.py --selected_model <1 of 4 models above> --audio_db_dir <path to database> --window_size <input sequence length> --forecast_size <output sequence length>

The audio database can be audio file in [".wav", ".aif", ".aiff", ".mp3", ".m4a"]


### Prediction (notebook)

The prediction workflow can be described in the diagram below:

![descriptor workflow](https://github.com/buganart/descriptor-transformer/blob/main/_static/img/descriptor_model_predict_workflow.png)

1. The prediction database will be processed into **descriptor input (descriptor database II)** for the descriptor model, and the descriptor model will *predict the subsequent descriptors* based on the input.
2. The audio database will be processed into **descriptor database I** that each descriptor will have *ID reference* back to the audio segment. 
3. The **query function** will replace the predicted new descriptors from the descriptor model with the closest match in the **descriptor database I** based on the distance function.
4. The audio segments referenced by the replaced descriptors from the query function will be combined and merged into a new audio file.

The [prediction notebook](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/descriptor_model_predict.ipynb) for the descriptor model is located in [predict_notebook/](https://github.com/buganart/descriptor-transformer/tree/main/predict_notebook).

Follow the instruction in the [prediction notebook](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/descriptor_model_predict.ipynb) to generate new descriptor and convert them back to audio.