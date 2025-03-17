import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os


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
        print(f'Best parameters: {best_params}')
        print(f'Validation Accuracy: {accuracy:.2f}')
    else:
        best_params = model.get_params()
        accuracy = None

    best_model = model.set_params(**best_params)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {test_accuracy:.2f}')
    cm = confusion_matrix(y_test, y_pred)
    save_results(result_dir, best_params, accuracy, test_accuracy, cm)
    return best_params


def evaluate_classic_models(processed_data_dir):
    cnn_train_path = os.path.join(processed_data_dir, 'cnn_train.csv')
    cnn_test_path = os.path.join(processed_data_dir, 'cnn_test.csv')
    resnet_train_path = os.path.join(processed_data_dir, 'resNet50_train.csv')
    resnet_test_path = os.path.join(processed_data_dir, 'resNet50_test.csv')

    cnn_X_train, cnn_y_train, cnn_X_test, cnn_y_test = load_data(cnn_train_path, cnn_test_path)
    resnet_X_train, resnet_y_train, resnet_X_test, resnet_y_test = load_data(resnet_train_path, resnet_test_path)

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

    # Обучение и оценка моделей для CNN данных
    print("Training and evaluating models for CNN data")
    rf_best_params = train_and_evaluate(RandomForestClassifier(random_state=42), rf_param_grid, cnn_X_train, cnn_y_train,
                                        cnn_X_test, cnn_y_test, '../result/CNN/RandomForest')
    gb_best_params = train_and_evaluate(GradientBoostingClassifier(random_state=42), gb_param_grid, cnn_X_train,
                                        cnn_y_train, cnn_X_test, cnn_y_test, '../result/CNN/GradientBoosting')
    svc_best_params = train_and_evaluate(SVC(random_state=42), svc_param_grid, cnn_X_train, cnn_y_train, cnn_X_test,
                                         cnn_y_test, '../result/CNN/SVC')
    sgd_best_params = train_and_evaluate(SGDClassifier(random_state=42), sgd_param_grid, cnn_X_train, cnn_y_train,
                                         cnn_X_test, cnn_y_test, '../result/CNN/SGDClassifier')
    gnb_best_params = train_and_evaluate(GaussianNB(), {}, cnn_X_train, cnn_y_train, cnn_X_test, cnn_y_test,
                                         '../result/CNN/GaussianNB', use_grid_search=False)

    # Обучение и оценка моделей для ResNet данных с лучшими параметрами от CNN
    print("Training and evaluating models for ResNet data with best CNN parameters")
    train_and_evaluate(RandomForestClassifier(random_state=42), rf_best_params, resnet_X_train, resnet_y_train,
                       resnet_X_test, resnet_y_test, '../result/ResNet/RandomForest_best_cnn', use_grid_search=False)
    train_and_evaluate(GradientBoostingClassifier(random_state=42), gb_best_params, resnet_X_train, resnet_y_train,
                       resnet_X_test, resnet_y_test, '../result/ResNet/GradientBoosting_best_cnn', use_grid_search=False)
    train_and_evaluate(SVC(random_state=42), svc_best_params, resnet_X_train, resnet_y_train, resnet_X_test,
                       resnet_y_test, '../result/ResNet/SVC_best_cnn', use_grid_search=False)
    train_and_evaluate(SGDClassifier(random_state=42), sgd_best_params, resnet_X_train, resnet_y_train, resnet_X_test,
                       resnet_y_test, '../result/ResNet/SGDClassifier_best_cnn', use_grid_search=False)
    train_and_evaluate(GaussianNB(), gnb_best_params, resnet_X_train, resnet_y_train, resnet_X_test, resnet_y_test,
                       '../result/ResNet/GaussianNB_best_cnn', use_grid_search=False)

    # Обучение и оценка моделей для ResNet данных со стандартными параметрами
    print("Training and evaluating models for ResNet data with default parameters")
    train_and_evaluate(RandomForestClassifier(random_state=42), rf_param_grid, resnet_X_train, resnet_y_train,
                       resnet_X_test, resnet_y_test, '../result/ResNet/RandomForest')
    train_and_evaluate(GradientBoostingClassifier(random_state=42), gb_param_grid, resnet_X_train, resnet_y_train,
                       resnet_X_test, resnet_y_test, '../result/ResNet/GradientBoosting')
    train_and_evaluate(SVC(random_state=42), svc_param_grid, resnet_X_train, resnet_y_train, resnet_X_test,
                       resnet_y_test, '../result/ResNet/SVC')
    train_and_evaluate(SGDClassifier(random_state=42), sgd_param_grid, resnet_X_train, resnet_y_train, resnet_X_test,
                       resnet_y_test, '../result/ResNet/SGDClassifier')
    train_and_evaluate(GaussianNB(), {}, resnet_X_train, resnet_y_train, resnet_X_test, resnet_y_test,
                       '../result/ResNet/GaussianNB', use_grid_search=False)
