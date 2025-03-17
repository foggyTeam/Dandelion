import os
from time import time

import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import preprocess_input


def colored_print(text: str, color: str = 'grey'):
    if color == 'grey':
        print(f"\033[90m{text}\033[0m")
    elif color == 'r':
        print(f"\033[91m{text}\033[0m")
    elif color == 'g':
        print(f"\033[92m{text}\033[0m")
    elif color == 'y':
        print(f"\033[93m{text}\033[0m")
    elif color == 'm':
        print(f"\033[95m{text}\033[0m")
    elif color == 'c':
        print(f"\033[96m{text}\033[0m")


def check_folders(raw_data_path, processed_data_path):
    if not os.path.exists(raw_data_path):
        os.mkdir(raw_data_path)
    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)


def save_csv(df, path, df_name):
    df.to_csv(os.path.join(path, df_name), index=False)

    colored_print(df_name + ' saved as csv.')


def extract_features(data_generator, model, resnet50=False):
    batches_count = len(data_generator)
    i = 0
    colored_print(f"{batches_count} batches are going to be processed.", 'y')

    # Extracting features:
    features = []
    labels = []

    start_time = time()
    for batch in data_generator:
        X, y = batch

        if resnet50:
            X = preprocess_input(X)

        feature = model.predict(X)
        feature_flattened = feature.reshape((feature.shape[0], -1))
        features.append(feature_flattened)
        labels.append(y)

        i += 1
        if i % 5 == 0:
            colored_print(f"Processed {int(i / batches_count * 100)}%: {i}/{batches_count}", 'y')
        if i == batches_count:
            break

    end_time = time()

    colored_print(f"Processed in {(end_time - start_time) / 60:.2f} minutes")

    X = np.concatenate(features, axis=0)
    y = np.concatenate([np.argmax(batch, axis=1) for batch in labels], axis=0)

    # Transforming 4D features into 2D array
    X_reshaped = X.reshape(X.shape[0], -1)

    joint_df = pd.DataFrame(X_reshaped)
    joint_df['flower_type'] = y
    return X, y, joint_df
