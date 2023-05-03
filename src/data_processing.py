from src import pd

def preprocess_data(file_path):
    """
    to do:
        inputs data raw , labels, time stamps
        returns processed data raw in floating point,the data must be centered at zero, labels, time_stamps (maybe changed or verified)
    """
    df = pd.read_csv(file_path)
    # To Do
    return df
