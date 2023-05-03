from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from src import pd
def evaluate_model(model, trained_model, df_test):
    """Evaluates a machine learning model and returns the evaluation metrics."""

    # Split features and classes
    X_test = df.iloc[:, :-1]
    y_test = df.iloc[:, -1:]

    # Metrics: 
    metrics = ['accuracy', 'precision', 'roc_auc', 'recall', 'f1_score', 'mean_absolute_error']

    # Model prediction
    y_pred = model.predict(X_test)

    # Dataframe metric results
    results = pd.DataFrame(index = [model], columns = metrics)

    # Calculate metrics
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['roc'] = roc_auc_score(y_test, y_pred, multi_class='ovr')
    results['recall'] = recall_score(y_test, y_pred)
    results['f1_score'] = f1_score(y_test, y_pred)
    results['mean_absolute_error'] = mean_absolute_error(y_test, y_pred)
    
    return results
