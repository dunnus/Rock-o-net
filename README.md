# Rock-o-net: LSTM based Rock Guitar Generator
Training a model which can learn the intricacies of a Lead guitar and is able to Generate its own composition from the training received

## Obtaining the data
We shall use this dataset of midi files as our source for this training.

### What is MIDI?
MIDI is short for Musical Instrument Digital Interface. Lets think of it as a way to store instrument data, wherein we can think of each song as something which contains one or more instruments (AKA tracks). Each track is said to contain one or more musical notes. Each note is an aggregation of:

1. Pitch of the note
2. Volume of the note
3. Time step relative to previous note

MIDI also contains a lot more data and also metadata related to the said data. But we don’t really care about all that for now.

### Parsing the data 
An important step of all Deep learning projects is ensuring that we parse the data and have good clean data for the model to learn on.

So our plan here is to:

1. Filter out only those MIDI files which have a ‘Guitar Instrument’ in them.
2. Get all notes form the filtered MIDI tracks for the ‘Guitar Instrument’.
3. Split these notes into batches of length sequence_length + 1.
4. Only pick sequences which have a minimum of unique_factor number of unique notes in them.
5. Now our X will be data[0:sequence_length] and Y will be data[sequence_length]
6. One hot encode Y. Normalize and Standardize X.

## Model Architecture

![](https://github.com/dunnus/Rock-o-net/blob/main/object/model.png)

This the model we’ll be using. Some notes here:

1. We’ll be using the Adam optimizer
2. Loss will be calculated with Categorical Cross Entropy. This is because our Y values are One-hot Encoded.
3. We will iterate this training over learning rates of 0.01, 0.001 and 0.0001.
4. We run 200 epochs for each given learning rate.

## Results

The loss against various learning rates are shown below

![](https://github.com/dunnus/Rock-o-net/blob/main/object/losschart.png)

## Usage
1. Generate the required X and Y for training by running python create_data.py. You can mess around in create_data.py to change the sequence_length and unique_factor parameters.
2. Modify the model as per your requirements in train.py. To train run python train.py.
3. Generate new music from random samples by running python generate.py.

##Generated Music


https://user-images.githubusercontent.com/73417038/182106363-1812542f-a0b5-488e-9ae4-27104455012c.mp4


https://user-images.githubusercontent.com/73417038/182106396-cfe6d703-9ee5-4779-86c8-b6506ac00d4f.mp4


https://user-images.githubusercontent.com/73417038/182106423-bf6746e3-1eb1-4f56-83ed-8e6a0a6bc9ce.mp4


https://user-images.githubusercontent.com/73417038/182106456-8285ca00-8f45-4939-9a37-76a7189eea61.mp4


## Credits
https://github.com/jisungk/deepjazz
https://github.com/Skuldur/Classical-Piano-Composer
