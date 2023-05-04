from src import RepeatedKFold, GridSearchCV, KNeighborsClassifier, SVC

def train_model(model, df):
    """Trains a machine learning model and returns the fitted model object."""

    # Split features and classes
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    # Repeated K-Fold cross validation - Stratified
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)

    # Metrics
    metrics = ['accuracy', 'precision']

    if model == 'KNN':

        # Model Pameters
        parameters_knn = {'n_neighbors': [5, 7, 9, 11, 13, 15], 'weights': ['uniform', 'distance'],
                          'metric': ['minkowski', 'euclidean', 'manhattan'], 'algorithm': ['kd_tree', 'brute']}

        # Model
        knn = KNeighborsClassifier()

        # Hyperparameters tuning - Grid Search
        grid_search_knn = GridSearchCV(
            knn, parameters_knn, scoring=metrics, n_jobs=-1, cv=rkf)
        grid_search_knn = grid_search_knn.fit(X, y)

        model = grid_search_knn.best_estimator_

    elif model == 'Support Vector Machine':

        # Model Pameters
        parameters_svm = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [
            1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'poly']}

        # Model
        svm = SVC(decision_function_shape='ovo')

        # Hyperparameters tuning - Grid Search
        grid_search_svm = GridSearchCV(
            svm, parameters_svm, scoring=metrics, n_jobs=-1, cv=rkf)
        grid_search_svm = grid_search_svm.fit(X, y)

        # Best model
        model = grid_search_svm.best_estimator_

    else:
        print('Unknown model')

    return model

