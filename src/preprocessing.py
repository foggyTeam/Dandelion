import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


def save_paths(target_dir):
    paths = []
    labels = []
    for flower in os.listdir(target_dir):
        flower_folder = os.path.join(target_dir, flower)
        for image in os.listdir(flower_folder):
            image_path = os.path.join(flower_folder, image)
            paths.append(image_path)
            labels.append(flower)
    return paths, labels


def collect_data(raw_data_dir):
    raw_data_train_dir = os.path.join(raw_data_dir, 'train')
    raw_data_test_dir = os.path.join(raw_data_dir, 'test')
    data = dict({})  # array contains all paths to images
    labels = dict({})  # array contains the target value of the image (i.e. flower name)

    data['train'], labels['train'] = save_paths(raw_data_train_dir)
    data['test'], labels['test'] = save_paths(raw_data_test_dir)

    return data, labels


def augment_normalize_data(data_paths, labels, batch_size=32, target_size=(224, 224)):
    transformed_train = None
