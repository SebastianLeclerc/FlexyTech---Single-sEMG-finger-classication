from src.data_processing import preprocess_data
from src.feature_extraction import extract_features
from src.model_training import train_model
from src.model_evaluation import evaluate_model

data_file_path = ''
processed_data = preprocess_data(data_file_path)

X, y = extract_features(processed_data)

models = ['', '', 'Support Vector Machine']
trained_models = {}
for model in models:
    trained_models[model] = train_model(model)

eveluated_models = {}
for model, trained_model in trained_models.items():
    eveluated_models[model] = evaluate_model(model, trained_model)

model_file_path = ''
# To Do: save the best model
