import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import colored_print


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return pd.concat([train, test], ignore_index=True)


def basic_analysis(df):
    colored_print("\nBasic data characteristics:", 'y')
    print(df.info())
    colored_print("Data statistics:", 'y')
    print(df.describe())


def visualization(df):
    colored_print("\nVisualization:", 'y')
    plt.figure(figsize=(12, 6))
    sns.histplot(df, kde=True)
    plt.title('Dispersion')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    plt.title('Box diagrams')
    plt.show()

    plt.figure(figsize=(12, 6))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


def detect_outliers(df):
    colored_print("\nDetecting outliers:", 'y')
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_filtered = df[~((df < (Q1 - 3 * IQR)) | (df > (Q3 + 3 * IQR))).any(axis=1)]
    colored_print(f"Lines before removing outliers: {df.shape[0]}")
    colored_print(f"Lines after removing outliers: {df_filtered.shape[0]}")
    return df_filtered


def analyze_data(processed_data_dir):
    colored_print("\nCNN data:", 'c')
    cnn_train_path = os.path.join(processed_data_dir, 'cnn_train.csv')
    cnn_test_path = os.path.join(processed_data_dir, 'cnn_test.csv')
    cnn_data = load_data(cnn_train_path, cnn_test_path)

    basic_analysis(cnn_data)
    # visualization(cnn_data)
    detect_outliers(cnn_data)

    colored_print("\nResNet50 data:", 'c')
    resnet_train_path = os.path.join(processed_data_dir, 'resNet50_train.csv')
    resnet_test_path = os.path.join(processed_data_dir, 'resNet50_test.csv')
    resnet_data = load_data(resnet_train_path, resnet_test_path)

    basic_analysis(resnet_data)
    # visualization(resnet_data)
    detect_outliers(resnet_data)
