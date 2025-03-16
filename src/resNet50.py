from time import time
import numpy as np
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model


def extract_features_resnet50(data_generator):
    # Initializing ResNet50:
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # pretrained model
    model = Model(inputs=base_model.input, outputs=base_model.output)

    batches_count = len(data_generator)
    i = 0
    print(f"\033[93m{batches_count} batches is going to be processed.\033[0m")

    # Extracting features:
    features = []
    labels = []

    start_time = time()
    for batch in data_generator:
        X, y = batch
        X = preprocess_input(X)
        feature = model.predict(X)
        features.append(feature)
        labels.append(y)

        i += 1
        if i % 5 == 0:
            print(f"\033[93mProcessed {int(i / batches_count * 100)}%: {i}/{batches_count}\033[0m")
        if i == batches_count:
            break
    end_time = time()

    print(f"\033[96mProcessed in {(end_time - start_time) / 60:.2f} minutes\033[0m")

    X = np.concatenate(features, axis=0)
    y = np.concatenate([np.argmax(batch, axis=1) for batch in labels], axis=0)

    # Transforming 4D features into 2D array
    X_reshaped = X.reshape(X.shape[0], -1)

    joint_df = pd.DataFrame(X_reshaped)
    joint_df['flower_type'] = y
    return X, y, joint_df
