from src import RepeatedKFold, GridSearchCV, KNeighborsClassifier, SVC, tf, kt, keras, Dense

def ann_model(hp):
    """
    Search the best hyperparameters combination and layer structure 
    :param hp: define the hyperparameters class
    """
    
    # Instance layered neural networks
    model = keras.Sequential()
    # First layer | input_dim = number of features
    model.add(Dense(100, input_dim=15, activation='relu'))
    
    # Search optimal number of layers
    for i in range(1, hp.Int("num_layers", 2, 6)):

        # Define dropout layer
        #model.add(keras.layers.Dropout(hp.Float("dropout_" + str(i), 0, 0.3, step=0.1)))

        model.add(
            keras.layers.Dense(
                # Number of neurons per layer 
                units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
                activation="relu")
            )
        
        # Define dropout layer
        model.add(keras.layers.Dropout(hp.Float("dropout_" + str(i), 0, 0.3, step=0.1)))
    
    # Output layer | Number neurons = number of classes
    model.add(Dense(3, activation='softmax'))
    
    # Define learning rate
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])
    
    return model

def train_model(model, X, y):
    """
    Trains a machine learning model and returns the fitted model object.
    :param model: ml model to be fitted
    :param X: features data
    :param y: labels
    """

    # Repeated K-Fold cross validation - Stratified
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)

    # Metric for evaluating hyperparameters tuning
    metrics = 'accuracy'

    if model == 'KNN':

        # Model Pameters
        parameters_knn = {'n_neighbors': list(range(1, 31)), 'weights': ['uniform', 'distance'],
                          'metric': ['minkowski', 'euclidean', 'manhattan'], 'algorithm': ['kd_tree', 'brute']}

        # Model
        knn = ()

        # Hyperparameters tuning - Grid Search
        grid_search_knn = GridSearchCV(
            knn, parameters_knn, scoring=metrics, n_jobs=-1, cv=rkf)
        grid_search_knn.fit(X, y)

        model = grid_search_knn

    elif model == 'Support Vector Machine':

        # Model Pameters
        parameters_svm = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf'], 'decision_function_shape':['ovo']}

        # Model
        svm = SVC()

        # Hyperparameters tuning - Grid Search
        grid_search_svm = GridSearchCV(
            svm, parameters_svm, scoring=metrics, n_jobs=-1, cv=rkf)
        grid_search_svm.fit(X, y)

        # Best model
        model = grid_search_svm
    
    elif model == 'ANN':

        # Apply one hot enconding in label is needed for multiclass classification
        dummy_y = tf.keras.utils.to_categorical(y,3)

        # Initialize keras tuner
        tuner = kt.Hyperband(ann_model,
                     objective="val_accuracy",
                     max_epochs=20,
                     factor=3,
                     overwrite=True,
                     hyperband_iterations=10,
                     directory="keras_tuner",
                     project_name="keras_tuner_hyperband",)

        # Stop early | Stop training when a monitored metric has stopped improving
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # Start the search and get the best model
        tuner.search(X, dummy_y, epochs=20, validation_split=0.2, callbacks=[stop_early], verbose=2)

        # Get the combination of best hyperparameters 
        best_hp=tuner.get_best_hyperparameters()[0]

        # Build model with best hyperparameters
        tuner_model = tuner.hypermodel.build(best_hp)

        # Train model 
        tuner_model.fit(X, dummy_y, epochs=100, validation_split=0.2, callbacks=[stop_early], verbose=2)
        
        model = tuner_model
        

    else:
        print('Unknown model')

    return model