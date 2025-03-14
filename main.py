from src.preprocessing import collect_data, augment_normalize_data
from src.resNet50 import extract_features_resnet50
from src.utils import save_csv

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
DATASET_NAME = 'National Flowers'


def main():
    print('Starting Dandelion!')

    # load and augment dataset
    data_paths, labels = collect_data(RAW_DATA_DIR)
    transformed_train, transformed_test = augment_normalize_data(data_paths, labels)

    # extract features from dataset
    X_train, y_train, train_df = extract_features_resnet50(transformed_train)
    X_test, y_test, test_df = extract_features_resnet50(transformed_test)

    # save extracted features
    save_csv(train_df, PROCESSED_DATA_DIR, 'train')
    save_csv(test_df, PROCESSED_DATA_DIR, 'test')

    # train models


if __name__ == '__main__':
    main()
