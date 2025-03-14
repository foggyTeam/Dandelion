from src.preprocessing import collect_data

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
DATASET_NAME = 'National Flowers'


def main():
    print('Starting Dandelion!')
    # load and augment dataset
    data_paths, labels = collect_data(RAW_DATA_DIR)

    # extract features from dataset

    # train models


if __name__ == '__main__':
    main()
