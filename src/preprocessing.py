import os

import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img


# python -c "import PIL;"

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


def add_padding(image, target_size):
    old_size = image.size
    ratio = float(target_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image.resize(new_size, Image.LANCZOS)
    new_image = Image.new("RGB", target_size)
    new_image.paste(image, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    return new_image


def augment_normalize_data(data_paths, labels, batch_size=64, target_size=(224, 224)):
    def preprocessing_function(image):
        image = array_to_img(image)
        image = add_padding(image, target_size)
        return img_to_array(image)

    # train might be normalized and augmented
    train_transformations = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rescale=1. / 255,  # Normalization
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest')

    # test can be normalized only
    test_transformations = ImageDataGenerator(preprocessing_function=preprocessing_function, rescale=1. / 255)

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
