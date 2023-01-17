# Adi Aviv, 206962904

import math
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from operator import itemgetter

class Recommender:
    def __init__(self, strategy='user'):
        self.strategy = strategy
        self.similarity = np.NaN

    def fit(self, matrix):
        self.user_item_matrix = matrix
        if self.strategy == 'user':
            # User - User based collaborative filtering
            start_time = time.time()
            mean_user_rate = np.round(np.nanmean(np.array(matrix), axis=1).reshape(-1, 1), decimals=5)
            ratings_diff = (np.array(matrix) - mean_user_rate + 0.001)
            ratings_diff[np.isnan(ratings_diff)] = 0
            self.bench = mean_user_rate
            self.bench_product = np.round(np.nanmean(np.array(matrix), axis=0), decimals=5)
            user_similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')
            self.pred = pd.DataFrame(mean_user_rate + (
                    user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T))
            self.pred = pd.DataFrame(self.pred).round(2)
            time_taken = time.time() - start_time
            print('User Model in {} seconds'.format(time_taken))
            return self

        elif self.strategy == 'item':
            # Item - Item based collaborative filtering
            start_time = time.time()
            mean_item_rate = np.nanmean(np.array(matrix), axis=1).reshape(-1, 1)
            ratings_diff = (np.array(matrix) - mean_item_rate)
            ratings_diff[np.isnan(ratings_diff)] = 0
            self.bench = mean_item_rate
            item_similarity = 1 - pairwise_distances(ratings_diff.T, metric='cosine')
            self.pred = pd.DataFrame(mean_item_rate + ratings_diff.dot(item_similarity) / np.array(
                [np.abs(item_similarity).sum(axis=1)]))
            self.pred = pd.DataFrame(self.pred).round(2)
            time_taken = time.time() - start_time
            print('Item Model in {} seconds'.format(time_taken))
            return self

    def find_indexes(self, idx, predicted_row, k):
        sim_scores = []
        i = 0
        while i < k:
            element = predicted_row[idx[i]]
            all_indexes = np.where(predicted_row == element)[0]
            sim_scores.extend(all_indexes)
            if len(sim_scores) > k:
                while len(sim_scores) > k:
                    sim_scores.pop()
                return sim_scores
            i += len(all_indexes)
        while len(sim_scores) > k:
            sim_scores.pop()
        return sim_scores

    def recommend_items(self, user_id, k=5):
        if user_id not in self.user_item_matrix.index:
            return None
        index = (self.user_item_matrix.index.get_loc(user_id))
        dta_matrix_row = self.user_item_matrix.iloc[index]
        x = np.where(dta_matrix_row.index == 'B0000533CC')
        predicted_row = np.array(self.pred)[index:index + 1]
        unrated = np.array(predicted_row[0])
        unrated[~np.isnan(dta_matrix_row)] = 0
        idx = np.argsort(-unrated)
        sim_scores = self.find_indexes(idx[:k], unrated, k)
        return dta_matrix_row.iloc[sim_scores].index.to_list()