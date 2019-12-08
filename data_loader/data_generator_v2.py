import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        ratings = pd.read_csv('/Users/apple/Desktop/ml-1m/ratings.dat', sep="::", header=None, engine='python')
        self.train_df, self.test_df = train_test_split(ratings,
                                             stratify=ratings[0],
                                             test_size=0.1,
                                             random_state=999613182)

        self.train_df, self.validate_df = train_test_split(self.train_df,
                                                 stratify=self.train_df[0],
                                                 test_size=0.1,
                                                 random_state=999613182)

        self.num_users = ratings[0].unique().max() + 1
        self.num_movies = ratings[1].unique().max() + 1
        print("loaded training data")

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.users_items_matrix_train_average[idx, :], self.users_items_matrix_train_zero[idx, :]

    def dataPreprocessor(self, rating_df, num_users, num_items, init_value=0, average=False):
        """
            INPUT:
                data: pandas DataFrame. columns=['userID', 'itemID', 'rating' ...]
                num_row: int. number of users
                num_col: int. number of items

            OUTPUT:
                matrix: 2D numpy array.
        """
        if average:
            matrix = np.full((num_users, num_items), 0.0)
            for (_, userID, itemID, rating, timestamp) in rating_df.itertuples():
                matrix[userID, itemID] = rating
            avergae = np.true_divide(matrix.sum(1), np.maximum((matrix != 0).sum(1), 1))
            inds = np.where(matrix == 0)
            matrix[inds] = np.take(avergae, inds[0])

        else:
            matrix = np.full((num_users, num_items), init_value)
            for (_, userID, itemID, rating, timestamp) in rating_df.itertuples():
                matrix[userID, itemID] = rating

        return matrix

    def generate_data(self):
        self.users_items_matrix_train_zero = self.dataPreprocessor(self.train_df, self.num_users, self.num_movies, 0)
        self.users_items_matrix_validate = self.dataPreprocessor(self.validate_df, self.num_users, self.num_movies, 0)
        self.users_items_matrix_test = self.dataPreprocessor(self.test_df, self.num_users, self.num_movies, 0)
        self.users_items_matrix_train_average = self.dataPreprocessor(self.train_df, self.num_users, self.num_movies, average=True)
        return self.users_items_matrix_train_average, self.users_items_matrix_validate, self.users_items_matrix_test
