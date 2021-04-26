# Descriptor Transformer Predict

Extract audio descriptors and learn to generate them with transformers.
The scripts here is used to train the descriptor model, melgan, and unagan.
While the descriptor model is necessary for the **descriptor prediction** and generate result audios, the melgan and unagan are optional and will be used for generating audio database for the **query function**.

NOTE: The melgan and unagan training is still not available to users that do not have access to wandb project repository **demiurge/melgan** and **demiurge/unagan**.

## Description of the prediction workflow

The prediction workflow can be described in the diagram below:

![descriptor workflow](https://github.com/buganart/descriptor-transformer/blob/main/_static/img/descriptor_model_predict_workflow.png)

1. The prediction database will be processed into **descriptor input (descriptor database II)** for the descriptor model, and the descriptor model will *predict the subsequent descriptors* based on the input.
2. The audio database will be processed into **descriptor database I** that each descriptor will have *ID reference* back to the audio segment. 
3. The **query function** will replace the predicted new descriptors from the descriptor model with the closest match in the **descriptor database I** based on the distance function.
4. The audio segments referenced by the replaced descriptors from the query function will be combined and merged into a new audio file.

### Descriptor train notebook

This notebook is used to train descriptor model, which is the crucial component in the **descriptor prediction** step. The descriptor model is a time-series prediction model that will predict subsequent descriptors given the input descriptors. 

In the notebook, there are 4 models to choose from: "LSTM", "LSTMEncoderDecoderModel", "TransformerEncoderOnlyModel", or "TransformerModel".
The "LSTM" and "TransformerEncoderOnlyModel" are one step prediction model, while "LSTMEncoderDecoderModel" and "TransformerModel" can predict descriptor sequence with specified sequence length.

### Optional: unagan notebook and melgan notebook

The unagan and melgan is for generating lots of audios to form audio database in case the user does not have their database for the descriptor model predict notebook. The [unagan generate notebook](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/Unagan_generate.ipynb) already has pretrained unagan melgan to generate audios. If users has their own small dataset, and they want to have more similar audios to fill the descriptor database space, then training a new unagan and melgan will be a good option to go for. The melgan and unagan is based on the github [descriptinc/melgan-neurips](https://github.com/descriptinc/melgan-neurips) and [ciaua/unagan](https://github.com/ciaua/unagan) repositories.

