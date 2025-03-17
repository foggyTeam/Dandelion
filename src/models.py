import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils import colored_print

matplotlib.use('TkAgg')


def load_data(train_path, test_path):
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
    plt.title(f'Confusion Matrix for Optimized Model')
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
    plt.close()


def train_and_evaluate(model, param_grid, X_train, y_train, X_test, y_test, result_dir, use_grid_search=True):
    if use_grid_search:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        accuracy = grid_search.best_score_
        colored_print(f'Best parameters: {best_params}', 'y')
        colored_print(f'Validation Accuracy: {accuracy:.2f}', 'y')
    else:
        best_params = model.get_params()
        accuracy = None

    best_model = model.set_params(**best_params)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    colored_print(f'Test Accuracy: {test_accuracy:.2f}', 'y')
    cm = confusion_matrix(y_test, y_pred)
    save_results(result_dir, best_params, accuracy, test_accuracy, cm)
    return best_params


def evaluate_classic_models(processed_data_dir, results_dir):
    cnn_train_path = os.path.join(processed_data_dir, 'cnn_train.csv')
    cnn_test_path = os.path.join(processed_data_dir, 'cnn_test.csv')
    resnet_train_path = os.path.join(processed_data_dir, 'resNet50_train.csv')
    resnet_test_path = os.path.join(processed_data_dir, 'resNet50_test.csv')

    cnn_X_train, cnn_y_train, cnn_X_test, cnn_y_test = load_data(cnn_train_path, cnn_test_path)
    colored_print('CNN data loaded...')
    resnet_X_train, resnet_y_train, resnet_X_test, resnet_y_test = load_data(resnet_train_path, resnet_test_path)
    colored_print('ResNet50 data loaded...')

    resnet_X_train, resnet_X_test = apply_pca(resnet_X_train, resnet_X_test)

    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None, 10, 20, 30],
        'criterion': ['gini', 'entropy']
    }

    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    svc_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }

    sgd_param_grid = {
        'alpha': [0.0001, 0.001, 0.01],
        'penalty': ['l2', 'l1', 'elasticnet']
    }

    # Train and score models on CNN data
    colored_print('Train and score models on CNN data', 'y')
    cnn_results = os.path.join(results_dir, 'CNN')

    colored_print('Random Forest...')
    rf_best_params = train_and_evaluate(RandomForestClassifier(random_state=42), rf_param_grid, cnn_X_train,
                                        cnn_y_train, cnn_X_test, cnn_y_test, os.path.join(cnn_results, 'RandomForest'))
    colored_print('Gradient Boosting...')
    gb_best_params = train_and_evaluate(GradientBoostingClassifier(random_state=42), gb_param_grid, cnn_X_train,
                                        cnn_y_train, cnn_X_test, cnn_y_test,
                                        os.path.join(cnn_results, 'GradientBoosting'))
    colored_print('SVC...')
    svc_best_params = train_and_evaluate(SVC(random_state=42), svc_param_grid, cnn_X_train, cnn_y_train, cnn_X_test,
                                         cnn_y_test, os.path.join(cnn_results, 'SVC'))
    colored_print('SGD Classifier...')
    sgd_best_params = train_and_evaluate(SGDClassifier(random_state=42), sgd_param_grid, cnn_X_train, cnn_y_train,
                                         cnn_X_test, cnn_y_test, os.path.join(cnn_results, 'SGDClassifier'))
    colored_print('Gaussian Naive Bias...')
    gnb_best_params = train_and_evaluate(GaussianNB(), {}, cnn_X_train, cnn_y_train, cnn_X_test, cnn_y_test,
                                         os.path.join(cnn_results, 'GaussianNB'), use_grid_search=False)

    # Train and score models on ResNet50 data + the best CNN params
    colored_print('Train and score models on ResNet50 data + the best CNN params', 'y')
    resnet_results = os.path.join(results_dir, 'ResNet/best_CNN_params')

    colored_print('Random Forest...')
    train_and_evaluate(RandomForestClassifier(random_state=42), rf_best_params, resnet_X_train, resnet_y_train,
                       resnet_X_test, resnet_y_test, os.path.join(resnet_results, 'RandomForest'),
                       use_grid_search=False)
    colored_print('Gradient Boosting...')
    train_and_evaluate(GradientBoostingClassifier(random_state=42), gb_best_params, resnet_X_train, resnet_y_train,
                       resnet_X_test, resnet_y_test, os.path.join(resnet_results, 'GradientBoosting'),
                       use_grid_search=False)
    colored_print('SVC...')
    train_and_evaluate(SVC(random_state=42), svc_best_params, resnet_X_train, resnet_y_train, resnet_X_test,
                       resnet_y_test, os.path.join(resnet_results, 'SVC'), use_grid_search=False)
    colored_print('SGD Classifier...')
    train_and_evaluate(SGDClassifier(random_state=42), sgd_best_params, resnet_X_train, resnet_y_train, resnet_X_test,
                       resnet_y_test, os.path.join(resnet_results, 'SGDClassifier'), use_grid_search=False)
    colored_print('Gaussian Naive Bias...')
    train_and_evaluate(GaussianNB(), gnb_best_params, resnet_X_train, resnet_y_train, resnet_X_test, resnet_y_test,
                       os.path.join(resnet_results, 'GaussianNB'), use_grid_search=False)

    # Train and score models on ResNet50 data + default params
    colored_print('Train and score models on ResNet50 data + default params', 'y')
    resnet_default_results = os.path.join(results_dir, 'ResNet/default_params')

    colored_print('Random Forest...')
    train_and_evaluate(RandomForestClassifier(random_state=42), rf_param_grid, resnet_X_train, resnet_y_train,
                       resnet_X_test, resnet_y_test, os.path.join(resnet_default_results, 'RandomForest'))
    colored_print('Gradient Boosting...')
    train_and_evaluate(GradientBoostingClassifier(random_state=42), gb_param_grid, resnet_X_train, resnet_y_train,
                       resnet_X_test, resnet_y_test, os.path.join(resnet_default_results, 'GradientBoosting'))
    colored_print('SVC...')
    train_and_evaluate(SVC(random_state=42), svc_param_grid, resnet_X_train, resnet_y_train, resnet_X_test,
                       resnet_y_test, os.path.join(resnet_default_results, 'SVC'))
    colored_print('SGD Classifier...')
    train_and_evaluate(SGDClassifier(random_state=42), sgd_param_grid, resnet_X_train, resnet_y_train, resnet_X_test,
                       resnet_y_test, os.path.join(resnet_default_results, 'SGDClassifier'))
    colored_print('Gaussian Naive Bias...')
    train_and_evaluate(GaussianNB(), {}, resnet_X_train, resnet_y_train, resnet_X_test, resnet_y_test,
                       os.path.join(resnet_default_results, 'GaussianNB'), use_grid_search=False)
