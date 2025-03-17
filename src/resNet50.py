import os

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.utils import extract_features, colored_print


def create_resnet50(input_shape, number_of_classes, tune_at_layer):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # saving from train all layers except 5 top layers
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[tune_at_layer:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(number_of_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # compiling a model
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def extract_features_resnet50(data_generator, raw_data_dir):
    raw_data_train_dir = os.path.join(raw_data_dir, 'train')
    input_shape = (224, 224, 3)
    tune_at_layer = 165  # 5 top layers
    number_of_classes = len(os.listdir(raw_data_train_dir))

    # Initializing ResNet50:
    colored_print("Initializing ResNet50...", 'y')
    model = create_resnet50(input_shape, number_of_classes, tune_at_layer)

    # Training the model on data
    colored_print("Training and validating model on data...", 'y')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    model.fit(data_generator['train'], validation_data=data_generator['test'], epochs=30, callbacks=[early_stopping])

    model_for_extraction = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)

    colored_print('\nExtracting from train:')
    X_train, y_train, train_df = extract_features(data_generator['train'], model_for_extraction, resnet50=True)
    colored_print('\nExtracting from test:')
    X_test, y_test, test_df = extract_features(data_generator['test'], model_for_extraction, resnet50=True)

    return X_train, y_train, train_df, X_test, y_test, test_df
