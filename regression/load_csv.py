import pandas as pd

def load_csv(filename: str, idx: str):
    """load csv file into dataframe, display two sample of df, print and return df"""
    train_data = pd.read_csv(filename,index_col=idx)
    train_data.sample(2)
    train_stats = train_data.describe().transpose()
    print(train_stats)
    return train_data