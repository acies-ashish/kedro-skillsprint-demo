import pandas as pd
from sklearn.model_selection import train_test_split

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and NA, ensure correct dtypes."""
    data = data.drop_duplicates()
    data = data.dropna()
    return data

def split_dataset(data, split_ratio, random_state, input_features, target):
    X = data[input_features]       # feature selection
    y = data[target]               # target selection
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=random_state, stratify=y
    )
    return X_train.assign(**{target: y_train}), X_test.assign(**{target: y_test})
