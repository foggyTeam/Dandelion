from src.preprocessing import collect_data, augment_normalize_data
from src.resNet50 import extract_features_resnet50
from src.utils import check_folders, save_csv

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
DATASET_NAME = 'National Flowers'


def main():
    print('\n\033[92mStarting Dandelion!\033[0m\n')

    # checking data folders existence
    check_folders(RAW_DATA_DIR, PROCESSED_DATA_DIR)

    # load and augment dataset
    data_paths, labels = collect_data(RAW_DATA_DIR)
    train_data_generator, test_data_generator = augment_normalize_data(data_paths, labels)

    # extract features from dataset via ResNet50
    print('\n\033[92mExtracting from train:\033[0m')
    X_train, y_train, train_df = extract_features_resnet50(train_data_generator)
    save_csv(train_df, PROCESSED_DATA_DIR, 'resNet50_train.csv')

    print('\n\033[92mExtracting from test:\033[0m')
    X_test, y_test, test_df = extract_features_resnet50(test_data_generator)
    save_csv(test_df, PROCESSED_DATA_DIR, 'resNet50_test.csv')

    # train models


if __name__ == '__main__':
    main()
