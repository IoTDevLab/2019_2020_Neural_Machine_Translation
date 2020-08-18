# Neural Machine Translator

This project can be used to train a translation model for any pair language.
Example Twi-to-English , English-to-Twi English-to-Ga 

# Dataset
1. Copy your dataset into the data directory
2. Set the FILE_PATH variable in the nmt_atten.ipynb notebook to the full path to your dataset
3. The source sentence and the target dataset must be in the same file seperated by tap.
4. Preferably save your dataset as a text file. See sample in the data directory


# Training
1. Set the EPOCHS variable in the nmt_atten.ipynb notebook to the number of epochs you want the model be trained on the dataset.
2. By default the EPOCHS variable in the nmt_atten.ipynb notebook has been set to 50 
3. After training the model will be saved to the models directory

# Testing
1. Your can test the model by loading it from the models directory



