from src import preprocess_data, train_model, evaluate_model, extract_features, pickle, os, time, train_test_split,combine_data
from feature_extraction import *








def main():
    # loading the and processing data.
    preprocessed_data,labels,timeStamps = preprocess_data("../../data")
    # parameters for feature extraction.
    features = [mean,median] # add the features.
    no_features= 2 # no.frames
    overlapping = 0.25 # overlapping window percentage. (0-1)
    # creating a dataframe with all the features for model training.
    df = extract_features(data=preprocessed_data,features_no=no_features,overlapping_percentage=overlapping,
                          features_funcs=features)
    # training model.
    model = train_model(model='..',df=df)
    # model evaluation
    #### to do.


if __name__ == '__main__':
    main









# data_file_path = ''
# processed_data = preprocess_data(data_file_path)

# # Split features and classes
# X = processed_data.iloc[:, :-1]
# y = processed_data.iloc.iloc[:, -1:]

# # Split data 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)

# # Train data
# df_train = pd.concat([X_train,y_train], axis=1)

# # Test data
# df_test = pd.concat([X_test,y_test], axis=1)

# X, y = extract_features(processed_data)

# models = ['', 'KNN', 'Support Vector Machine']
# trained_models = {}
# # Dataframe which store metrics for each model
# eval_results = pd.DataFrame()
# for model in models:
#     trained_models[model] = train_model(model, df_train)

# for model, trained_model in trained_models.items():
#     eval_results = pd.concat([eval_results, evaluate_model(model, trained_model, df_test)], axis=1)

# # Select best model based on X metric 
# best_model = trained_models[str(eval_results['metric_selected'].idxmax())]
# model_filename = f"{best_model}_{time.strftime('%Y-%m-%d')}.pkl"
# model_file_path = os.path.join("../models", model_filename)
# with open(model_file_path, 'wb') as f:
#     pickle.dump(best_model, f)
