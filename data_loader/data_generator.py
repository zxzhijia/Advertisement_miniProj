import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        ratings = pd.read_csv('/Users/apple/Desktop/ml-1m/ratings.dat', sep="::", header=None, engine='python')
        ratings_pivot = pd.pivot_table(ratings[[0, 1, 2]], values=2, index=0, columns=1).fillna(0)
        self.input, self.X_test = train_test_split(ratings_pivot, train_size=0.8)
        print("loaded training data")

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input.iloc[idx], self.input.iloc[idx]
