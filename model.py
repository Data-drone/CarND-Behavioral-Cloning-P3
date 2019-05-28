"""

building a model for testing

training input: central image
training target: movement pedals

"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense, Conv2D

def nvidia_paper_model():

    """

    model definition from the nvidia behavioural modelling paper:


    """

    adv_model = Sequential()
    adv_model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    adv_model.add(Cropping2D(cropping=((70,25),(0,0))))
    
    adv_model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    adv_model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    adv_model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    adv_model.add(Conv2D(64, (3, 3), activation="relu"))
    adv_model.add(Conv2D(64, (3, 3), activation="relu"))
    adv_model.add(Flatten())
    adv_model.add(Dense(100))
    adv_model.add(Dense(50))
    adv_model.add(Dense(10))
    adv_model.add(Dense(1))

    return adv_model


def image_generator(data_table, batch_size=32):

    """

    reads in the processed car logs and yields batches of e and y
    
    """
    data_dir = 'data'

    while True:

        shuffle = data_table.sample(frac=1)

        x_set = []
        y_set = []

        for rn, row in shuffle.iterrows():

            path = os.path.join('.', data_dir, row['folder'], 'IMG', row['center_image'])
            image = plt.imread(path)
            x_set.append(image)
            y_set.append(row['steering_angle'])

            if len(x_set) == batch_size:
                
                x = np.array(x_set)
                y = np.array(y_set)

                yield (x,y)

                x_set = []
                y_set = []

        x = np.array(x_set)
        y = np.array(y_set)

        assert len(x) <= batch_size

        yield (x,y)


if __name__ == '__main__':

    # Load the processed dataset
    # drive logs assembled via the Explore_Data.ipynb notebook

    # location of datafiles
    data_dir = 'data'
    processed_data = 'processed_data'

    dataset = pd.read_csv(os.path.join(processed_data, 'dataset.csv'))

    # create image generator
    img_gen = image_generator(dataset)

    # build model
    model = nvidia_paper_model()

    model.compile(loss='mse', optimizer='adam')
    # set number of epochs based on size of dataset / batch_sixe
    epochs = round(len(dataset)/32)

    # fit model
    model.fit_generator(img_gen, steps_per_epoch=epochs, epochs=5) 

    # save model
    model.save('itr_model.h5')






