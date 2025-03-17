from src.cnn import extract_features_cnn
from src.kNN import evaluate_knn
from src.models import evaluate_classic_models
from src.preprocessing import collect_data, augment_normalize_data
from src.resNet50 import extract_features_resnet50
from src.utils import check_folders, save_csv, colored_print

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
RESULTS_DIR = 'result'
DATASET_NAME = 'National Flowers'


def main():
    colored_print('\nStarting Dandelion!\n', 'g')

    # checking data folders existence
    check_folders(RAW_DATA_DIR, PROCESSED_DATA_DIR)

    # extract features from images
    colored_print('Do you need to extract features from dataset? y / n', 'c')
    if 'y' in input():
        # load and augment dataset
        data_paths, labels = collect_data(RAW_DATA_DIR)
        train_data_generator, test_data_generator = augment_normalize_data(data_paths, labels)

        colored_print('ResNet50? y / n', 'c')
        if 'y' in input():
            # extract features from dataset via ResNet50
            colored_print('\nExtracting from train:')
            X_train, y_train, train_df = extract_features_resnet50(train_data_generator)
            save_csv(train_df, PROCESSED_DATA_DIR, 'resNet50_train.csv')

            colored_print('\nExtracting from test:')
            X_test, y_test, test_df = extract_features_resnet50(test_data_generator)
            save_csv(test_df, PROCESSED_DATA_DIR, 'resNet50_test.csv')

        colored_print('Custom CNN? y / n', 'c')
        if 'y' in input():
            # extract features from dataset via custom CNN
            colored_print('\nExtracting from train:')
            X_train, y_train, train_df = extract_features_cnn(train_data_generator, RAW_DATA_DIR)
            save_csv(train_df, PROCESSED_DATA_DIR, 'cnn_train.csv')

            colored_print('\nExtracting from test:')
            X_test, y_test, test_df = extract_features_cnn(test_data_generator, RAW_DATA_DIR)
            save_csv(test_df, PROCESSED_DATA_DIR, 'cnn_test.csv')

    # train models
    colored_print('Do you need to train models? y / n', 'c')
    if 'y' in input():
        colored_print('kNN? y / n', 'c')
        if 'y' in input():
            evaluate_knn(PROCESSED_DATA_DIR, RESULTS_DIR)

        colored_print('Classic models? y / n', 'c')
        if 'y' in input():
            evaluate_classic_models(PROCESSED_DATA_DIR, RESULTS_DIR)


if __name__ == '__main__':
    main()
