import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


def augment_normalize_data(data_paths, labels, batch_size=64, target_size=(224, 224)):
    # train might be normalized and augmented
    train_transformations = ImageDataGenerator(
        rescale=1. / 255,  # Normalization
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # test can be normalized only
    test_transformations = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_transformations.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': data_paths['train'], 'class': labels['train']}),
        x_col='filename',
        y_col='class',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_transformations.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': data_paths['test'], 'class': labels['test']}),
        x_col='filename',
        y_col='class',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator
