# Descriptor Transformer Predict

Extract audio descriptors and learn to generate them with transformers.
After the descriptor model is trained with the [train notebook](https://github.com/buganart/descriptor-transformer/blob/main/train_notebook/descriptor_model_train.ipynb), the scripts here will use those models to generate new descriptors and combine them into audio.

## Descriptor model predict notebook

The prediction workflow can be described in the diagram below:

![descriptor workflow](https://github.com/buganart/descriptor-transformer/blob/main/_static/img/descriptor_model_predict_workflow.png)

1. The prediction database will be processed into **descriptor input (descriptor database II)** for the descriptor model, and the descriptor model will *predict the subsequent descriptors* based on the input.
2. The audio database will be processed into **descriptor database I** that each descriptor will have *ID reference* back to the audio segment. 
3. The **query function** will replace the predicted new descriptors from the descriptor model with the closest match in the **descriptor database I** based on the distance function.
4. The audio segments referenced by the replaced descriptors from the query function will be combined and merged into a new audio file.

### Optional: unagan generate notebook

The unagan generate notebook is for generating lots of audios to form audio database in case the user does not have their database for the descriptor model predict notebook. The notebook has pretrained unagan to generate mel spectrogram and pretrained melgan to convert the spectrogram back to audios. The unagan and melgan are trained based on our custom dataset. 

~~If the user want to train melgan and unagan with their own dataset, please go to the [melgan notebook](https://github.com/buganart/descriptor-transformer/blob/main/train_notebook/melgan.ipynb) and [unagan notebook](https://github.com/buganart/descriptor-transformer/blob/main/train_notebook/unagan.ipynb).~~

The melgan and unagan training is still not available to users that do not have access to wandb project repository **demiurge/melgan** and **demiurge/unagan**.

