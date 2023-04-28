from src import pd

def preprocess_data(file_path):
    """Loads data from file and returns processed data."""
    df = pd.read_csv(file_path)
    # To Do
    return df
