# Eden Cohen 318758778

import time
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import pairwise_distances


class Recommender:
    def __init__(self, strategy='user'):
        self.benchmark_user = None
        self.benchmark_product = None
        self.user_item_matrix = None
        self.strategy = strategy
        self.similarity = np.NaN

    def fit(self, matrix):
        self.user_item_matrix = matrix
        # calculate the average of the ranking
        mean_user_rating = np.nanmean(matrix, axis=1).reshape(-1, 1)
        self.benchmark_user = mean_user_rating
        self.benchmark_product = np.nanmean(matrix, axis=0).reshape(-1, 1)
        ratings_diff = (np.array(matrix) - mean_user_rating) + 0.001
        # replace nan -> 0.001
        ratings_diff[np.isnan(ratings_diff)] = 0

        if self.strategy == 'user':
            # User - User based collaborative filtering
            start_time = time.time()
            # find the similarity between users
            self.similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')
            print(self.similarity.shape)

            # For each user (i.e., for each row) keep only k most similar users, set the rest to 0.
            # Note that the user has the highest similarity to themselves.

            self.pred = pd.DataFrame(mean_user_rating + (
                    self.similarity.dot(ratings_diff) / np.array([np.abs(self.similarity).sum(axis=1)]).T))
            self.pred = pd.DataFrame(self.pred).round(2)
            time_taken = time.time() - start_time
            print('User Model in {} seconds'.format(time_taken))

            return self

        elif self.strategy == 'item':
            # Item - Item based collaborative filtering
            start_time = time.time()
            # calculate user x user similarity matrix
            self.similarity = 1 - pairwise_distances(ratings_diff.T, metric='cosine')
            self.pred = pd.DataFrame(mean_user_rating + ratings_diff.dot(self.similarity) / np.array(
                [np.abs(self.similarity).sum(axis=1)]))
            self.pred = pd.DataFrame(self.pred).round(2)
            time_taken = time.time() - start_time
            print('Item Model in {} seconds'.format(time_taken))

            return self

    def get_sim_scores(self, idx, k, predicted_data):
        scores = []
        for i in range(k):
            # find all indexes of the higher rate
            indexes = np.where(predicted_data == predicted_data[idx[i]])
            list_idx = indexes[0].tolist()
            list_idx.sort()
            for j in list_idx:
                if j not in scores:
                    scores.append(j)
        return scores[:k]

    def recommend_items(self, user_id, k=5):
        # check if the id exists
        if user_id not in self.user_item_matrix.index:
            return None
        # get row of the requested user
        user_row_index = self.user_item_matrix.index.get_loc(user_id)

        # get user ranking
        data = self.user_item_matrix.iloc[user_row_index]
        predicted_data = self.pred.iloc[user_row_index]

        # initiate items that the user already rated
        init_predicted_data = np.array(predicted_data)
        init_predicted_data[~np.isnan(data)] = 0
        sorted = np.sort(-init_predicted_data)
        idx = np.argsort(-init_predicted_data)
        # print (idx)
        sim_scores = self.get_sim_scores(idx[:k], k, init_predicted_data)

        # Return top k movies
        return data.iloc[sim_scores].index.to_list()
