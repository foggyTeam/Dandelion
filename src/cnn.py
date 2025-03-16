import os

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Sequential

from src.utils import extract_features


def create_cnn(input_shape, number_of_classes):
    model = Sequential()

    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(number_of_classes, activation='softmax'))

    return model


def extract_features_cnn(data_generator, raw_data_dir):
    raw_data_train_dir = os.path.join(raw_data_dir, 'train')
    number_of_classes = len(os.listdir(raw_data_train_dir))

    model = create_cnn((224, 224, 3), number_of_classes)

    return extract_features(data_generator, model)
