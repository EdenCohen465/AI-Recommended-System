# Eden Cohen 318758778

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def RMSE(test_set, cf):
    rmse_test_set = RMSE_test_set(test_set, cf)
    rmse_bench = RMSE_benchmark(test_set, cf)

    print(f"RMSE benchmark: {rmse_bench}")

    print(f"RMSE {cf.strategy} based: {rmse_test_set}")


def RMSE_test_set(test_set, cf):
    prediction_matrix = cf.pred
    rmse = 0

    for index, row in test_set.iterrows():
        user_id = row['UserId']
        actual_rating = row['Rating']
        product_id = row['ProductId']

        # find predicted rating
        user_index = cf.user_item_matrix.index.get_loc(user_id)
        product_index = cf.user_item_matrix.columns.get_loc(product_id)

        predicted_user_rating = prediction_matrix.iloc[user_index, product_index]

        rmse += (predicted_user_rating - actual_rating) ** 2

    final = np.sqrt(rmse / len(test_set))
    return round(final, 5)


def RMSE_benchmark(test_set, cf):
    benchmark = cf.benchmark_user
    rmse = 0
    for index, row in test_set.iterrows():
        user_id = row['UserId']
        actual_rating = row['Rating']

        # find predicted rating
        user_index = cf.user_item_matrix.index.get_loc(user_id)
        avg_user_rating = benchmark[user_index]

        rmse += (avg_user_rating - actual_rating) ** 2

    final = np.sqrt(rmse / len(test_set))
    return round(final[0], 5)


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def precision_at_k(test_set, cf, k):
    sum_pred = 0
    sum_bench = 0
    none_items = 0
    # get best k products - best average rating
    average_rating = cf.user_item_matrix.mean(axis=0)
    top_k_items_bench = average_rating.sort_values(ascending=False)[:k]

    copy_test = test_set.copy()
    copy_test = copy_test[copy_test['Rating'] >= 3]

    for user_id in test_set['UserId'].unique():
        # get k recommended prediction items for user
        recommended_items = cf.recommend_items(user_id, k)
        # get best k rated items from test set
        top_k_items_test = copy_test[copy_test['UserId'] == user_id].sort_values(by='Rating', ascending=False)[:k]['ProductId'].to_list()

        if len(top_k_items_test) == 0:
            none_items += 1
            continue

        len_intersection_pred = len(intersection(recommended_items, top_k_items_test))
        len_intersection_bench = len(intersection(top_k_items_bench.index, top_k_items_test))

        sum_pred += (len_intersection_pred / k)
        sum_bench += (len_intersection_bench / k)

    precision_pred = round(sum_pred / (len(test_set['UserId'].unique()) - none_items), 5)
    precision_bench = round(sum_bench / (len(test_set['UserId'].unique()) - none_items), 5)

    print(f"precision user based cf: {precision_pred}")
    print(f"precision benchmark: {precision_bench}")
    return precision_pred


def recall_at_k(test_set, cf, k):
    sum_pred = 0
    sum_bench = 0
    num_of_none_relevent = 0

    # get best k products - best average rating
    average_rating = cf.user_item_matrix.mean(axis=0)
    top_k_items_bench = average_rating.sort_values(ascending=False)[:k]

    copy_test = test_set.copy()
    copy_test = copy_test[copy_test['Rating'] >= 3]

    for user_id in test_set['UserId'].unique():
        # get k recommended prediction items for user
        recommended_items_pred = cf.recommend_items(user_id, k)

        # get best k rated items from test set
        user_ranking = copy_test[copy_test['UserId'] == user_id]
        top_k_test = user_ranking.sort_values(by='Rating', ascending=False)[:k]['ProductId'].to_list()
        if len(top_k_test) == 0:
            num_of_none_relevent += 1
            continue

        len_intersection_list = len(intersection(recommended_items_pred, top_k_test))
        len_intersection_bench = len(intersection(top_k_items_bench.index, top_k_test))

        sum_pred += (len_intersection_list / len(top_k_test))
        sum_bench += (len_intersection_bench / len(top_k_test))

    recall_pred = round(sum_pred / (len(test_set['UserId'].unique()) - num_of_none_relevent), 5)
    recall_bench = round(sum_bench / (len(test_set['UserId'].unique()) - num_of_none_relevent), 5)
    print(f"recall user based cf: {recall_pred}")
    print(f"recall benchmark: {recall_bench}")
    return recall_pred

    pass
