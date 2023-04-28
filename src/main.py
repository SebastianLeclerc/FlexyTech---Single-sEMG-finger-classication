from src import preprocess_data, train_model, evaluate_model, extract_features, pickle, os, time

data_file_path = ''
processed_data = preprocess_data(data_file_path)

X, y = extract_features(processed_data)

models = ['', '', 'Support Vector Machine']
trained_models = {}
for model in models:
    trained_models[model] = train_model(model)

for model, trained_model in trained_models.items():
    best_model = evaluate_model(model, trained_model)

model_filename = f"{best_model}_{time.strftime('%Y-%m-%d')}.pkl"
model_file_path = os.path.join("../models", model_filename)
with open(model_file_path, 'wb') as f:
    pickle.dump(best_model, f)