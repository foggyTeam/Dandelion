import numpy as np
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model


def extract_features_resnet50(transformed_data):
    # Initializing ResNet50:
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # pretrained model
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Extracting features:
    features = []
    labels = []

    for image in transformed_data:
        X, y = image
        X = preprocess_input(X)
        feature = model.predict(X)
        features.append(feature)
        labels.append(y)

    X, y = np.vstack(features), np.hstack(labels)
    joint_df = pd.DataFrame(X)
    joint_df['flower_type'] = y
    return X, y, joint_df
