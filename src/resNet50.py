from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

from src.utils import extract_features


def extract_features_resnet50(data_generator):
    # Initializing ResNet50:
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # pretrained model
    model = Model(inputs=base_model.input, outputs=base_model.output)

    return extract_features(data_generator, model, resnet50=True)
