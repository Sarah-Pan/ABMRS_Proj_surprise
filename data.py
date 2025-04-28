import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ratings_raw = pd.read_csv('ml-latest-small/ratings.csv')   # 包含 userId, movieId, rating, timestamp
# movies_raw = pd.read_csv('ml-latest-small/movies.csv')     # 包含 movieId, title, genres


# # ratings dataframe
# ratings = ratings_raw.rename(columns={
#     'userId': 'user_id',
#     'movieId': 'item_id',
#     'rating': 'rating'
# })
# ratings['consumption_time'] = pd.to_datetime(ratings_raw['timestamp'], unit='s')

# # items dataframe
# items = movies_raw.rename(columns={
#     'movieId': 'item_id',
#     'title': 'item_name'
# })[['item_id', 'item_name']]

# # users dataframe from rating.csv
# users = pd.DataFrame({'user_id': ratings['user_id'].unique()})
# users['user_name'] = users['user_id'].apply(lambda x: f'User {x}')

# Item data (replace with your own)

items = pd.DataFrame({'item_id': [1, 2, 3, 4, 5],
                     'item_name': ['Item A', 'Item B', 'Item C', 'Item D', 'Item E']})

# User data (replace with your own)
users = pd.DataFrame({'user_id': [1, 2, 3, 4, 5],
                     'user_name': ['User A', 'User B', 'User C', 'User D', 'User E']})

# Rating data (replace with your own)
ratings = pd.DataFrame({'user_id': [1, 1, 2, 2, 3],
                       'item_id': [1, 2, 2, 3, 1],
                       'rating': [4, 5, 3, 4, 2]})

ratings['consumption_time'] = [datetime.now() - timedelta(days= random.randint(1, 10)) for _ in range(len(ratings))]

