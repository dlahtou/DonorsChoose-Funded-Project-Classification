import logging
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

random_state = 10
random_iters = 20

def generate_lr_model(X_train, y_train):
    logging.info('Training Logistic Regression model')

    lr = LogisticRegression(solver='saga', n_jobs=-1)

    # Create param grid
    params = {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-3,1,5)
    }

    # Create estimator
    estimator = GridSearchCV(lr, params, scoring='roc_auc')

    # Train estimator
    estimator.fit(X_train, y_train)

    return estimator

def generate_gb_model(X_train, y_train):
    logging.info('Training GB model')

    gbc = GradientBoostingClassifier(random_state=random_state)

    # Create param grid
    params = {
        'subsample': [0.8],
        'learning_rate': np.logspace(-3, -1, 3),
        'n_estimators': [150,200,300,500],
        'max_depth': [2], #[3,4,5,6]
        'max_features': ['sqrt'],
        'min_samples_split': [10, 50, 100, 200, 300],
        'min_weight_fraction_leaf': [0.0001, 0.0005, 0.001, 0.005]
    }

    # Create estimator
    estimator = RandomizedSearchCV(gbc, params, n_iter=5, scoring='roc_auc', cv=5, random_state=10)

    # Train estimator
    estimator.fit(X_train, y_train)

    return estimator

def generate_rf_model(X_train, y_train):
    logging.info('Training Random Forest model')

    rfc = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    # Create param grid
    params = {
        'n_estimators': [150,200,300,500],
        'max_depth': [3,4,5,6],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [10, 50, 100, 200, 300]
    }

    # Create estimator
    estimator = RandomizedSearchCV(rfc, params, n_iter=40)

    # Train estimator
    estimator.fit(X_train, y_train)

    return estimator

def generate_knn_model(X_train, y_train):
    logging.info('Training KNN model')

    knn = KNeighborsClassifier(n_jobs=-1)

    # Create param grid
    params = {
        'n_neighbors': [5,10,25,50,125,200]
    }

    # Create estimator
    estimator = GridSearchCV(knn, params)

    # Train estimator
    estimator.fit(X_train, y_train)

    return estimator

def generate_svm_model(X_train, y_train):
    logging.info('Training svm model')

    # Create pipeline
    svm = SVC(probability=True)

    # Create param grid
    params = {
        'kernel': ['linear', 'rbf'],
        'C': [1,4,5,6,7,8,10]
    }

    # Create estimator
    estimator = RandomizedSearchCV(svm, params, n_iter=5)

    # Train estimator
    estimator.fit(X_train, y_train)

    return estimator