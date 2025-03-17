import matplotlib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os


def load_and_standardize_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def apply_pca(X_train, X_test, n_components=0.95):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


def save_results(result_dir, best_params, accuracy, test_accuracy, cm):
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'best_params_and_accuracy.txt'), 'w') as f:
        f.write(f'Best parameters: {best_params}\n')
        if accuracy is not None:
            f.write(f'Validation Accuracy: {accuracy:.2f}\n')
        f.write(f'Test Accuracy: {test_accuracy:.2f}\n')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix')
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
    plt.close()


def train_and_evaluate(model, param_grid, X_train, y_train, X_test, y_test, result_dir, use_grid_search=True):
    if use_grid_search:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        accuracy = grid_search.best_score_
    else:
        best_params = param_grid
        accuracy = None

    best_model = model.set_params(**best_params)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    save_results(result_dir, best_params, accuracy, test_accuracy, cm)
    return best_params


def evaluate_knn(processed_data_dir):
    cnn_train_path = os.path.join(processed_data_dir, 'cnn_train.csv')
    cnn_test_path = os.path.join(processed_data_dir, 'cnn_test.csv')
    resnet_train_path = os.path.join(processed_data_dir, 'resNet50_train.csv')
    resnet_test_path = os.path.join(processed_data_dir, 'resNet50_test.csv')

    cnn_X_train, cnn_y_train, cnn_X_test, cnn_y_test = load_and_standardize_data(cnn_train_path, cnn_test_path)
    resnet_X_train, resnet_y_train, resnet_X_test, resnet_y_test = load_and_standardize_data(resnet_train_path,
                                                                                             resnet_test_path)

    cnn_X_train_pca, cnn_X_test_pca = apply_pca(cnn_X_train, cnn_X_test)
    resnet_X_train_pca, resnet_X_test_pca = apply_pca(resnet_X_train, resnet_X_test)

    knn_param_grid = {
        'n_neighbors': range(1, 31),
        'metric': ['euclidean', 'manhattan']
    }

    # Обучение и оценка моделей для CNN данных
    knn_best_params = train_and_evaluate(KNeighborsClassifier(), knn_param_grid, cnn_X_train_pca, cnn_y_train,
                                         cnn_X_test_pca, cnn_y_test, '../result/CNN/kNN')

    # Обучение и оценка моделей для ResNet данных с лучшими параметрами от CNN
    best_knn_params_dict = {'n_neighbors': knn_best_params['n_neighbors'], 'metric': knn_best_params['metric']}
    train_and_evaluate(KNeighborsClassifier(), best_knn_params_dict, resnet_X_train_pca, resnet_y_train,
                       resnet_X_test_pca,
                       resnet_y_test, '../result/ResNet/kNN_best_cnn', use_grid_search=False)

    # Обучение и оценка моделей для ResNet данных со стандартными параметрами
    print("Training and evaluating model for ResNet data with default parameters")
    train_and_evaluate(KNeighborsClassifier(), knn_param_grid, resnet_X_train, resnet_y_train, resnet_X_test,
                       resnet_y_test, '../result/ResNet/kNN')
