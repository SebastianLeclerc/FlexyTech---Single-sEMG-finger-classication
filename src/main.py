from src import preprocess_data, train_model, evaluate_model, extract_features, pickle, os, time

data_file_path = ''
processed_data = preprocess_data(data_file_path)

# Split features and classes
X = processed_data.iloc[:, :-1]
y = processed_data.iloc.iloc[:, -1:]

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)

# Train data
df_train = pd.concat([X_train,y_train], axis=1)

# Test data
df_test = pd.concat([X_test,y_test], axis=1)

X, y = extract_features(processed_data)

models = ['', 'KNN', 'Support Vector Machine']
trained_models = {}
for model in models:
    trained_models[model] = train_model(model, df_train)

for model, trained_model in trained_models.items():
    best_model = evaluate_model(model, trained_model)

model_filename = f"{best_model}_{time.strftime('%Y-%m-%d')}.pkl"
model_file_path = os.path.join("../models", model_filename)
with open(model_file_path, 'wb') as f:
    pickle.dump(best_model, f)
