import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from data import items, users, ratings
from sklearn.metrics.pairwise import cosine_similarity 
from surprise import Dataset, Reader, KNNBasic
from surprise import PredictionImpossible



class Recommender:

    # Global variables for populations 
    item_population = {}  # average item rating
    user_population = {}  # average rating by the user

    def __init__(self, ratings):
        self.ratings = ratings
        self.update_populations()
        self.build_user_item_matrix()
        self.build_surprise_model()

    def build_surprise_model(self):
    # change dataframe to surprise format
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings[['user_id', 'item_id', 'rating']], reader)
        self.trainset = data.build_full_trainset()

        sim_options = {
            'name': 'cosine',     # cosine similarity
            'user_based': True    # True = user-based CF, False = item-based CF
        }

        self.surprise_model = KNNBasic(sim_options=sim_options)
        self.surprise_model.fit(self.trainset)

    def update_populations(self):
        # calculate items mean ratings
        item_group = self.ratings.groupby('item_id')['rating'].mean()
        Recommender.item_population = item_group.to_dict()

        # calculate users mean ratings
        user_group = self.ratings.groupby('user_id')['rating'].mean()
        Recommender.user_population = user_group.to_dict()

    def build_user_item_matrix(self):
        self.user_item_matrix = self.ratings.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(self.user_similarity,
                                               index=self.user_item_matrix.index,
                                               columns=self.user_item_matrix.index)

    def update_consumption(self, user_id, item_id, rating):
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        if user_ratings.empty:
            avg_interval = random.randint(1, 7)
        else:
            user_ratings = user_ratings.sort_values(by='consumption_time')
            intervals = user_ratings['consumption_time'].diff().dt.total_seconds() / (60 * 60 * 24)
            avg_interval = intervals.mean() if not intervals.empty else 3

        next_interval = max(1, np.random.normal(loc=avg_interval, scale=1))
        consumption_time = datetime.now() + timedelta(days=next_interval)

        formatted_time = consumption_time.strftime("%Y-%m-%d %H:%M")

        new_row = {
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'consumption_time': consumption_time
        }

        self.ratings = pd.concat([self.ratings, pd.DataFrame([new_row])], ignore_index=True)
        self.update_populations()
        self.build_user_item_matrix()
        print(f"[INFO] User {user_id} consumed Item {item_id} with rating {rating} at {formatted_time}")

    def calculate_rating_tendency(self, user_id, item_id):
        item_avg_rating = Recommender.item_population.get(item_id, 0)
        user_avg_rating = Recommender.user_population.get(user_id, 0)
        rating_tendency = (item_avg_rating + user_avg_rating) / 2
        return rating_tendency

    def predict_rating(self, user_id, item_id, k=5):
        try:
            prediction = self.surprise_model.predict(user_id, item_id)
            return prediction.est
        except PredictionImpossible:
            return self.calculate_rating_tendency(user_id, item_id)

    def decide_to_consume(self, user_id, item_id, predicted_rating):
        rating_tendency = self.calculate_rating_tendency(user_id, item_id)
        combined_rating = (predicted_rating + rating_tendency) / 2
        consumption_probability = combined_rating / 5.0
        return np.random.random() < consumption_probability

    def consume_item(self, user_id, item_id, predicted_rating=None):
        if predicted_rating is None:
            predicted_rating = self.predict_rating(user_id, item_id)

        if self.decide_to_consume(user_id, item_id, predicted_rating):
            rating_tendency = self.calculate_rating_tendency(user_id, item_id)
            rating = round(np.random.normal(loc=rating_tendency, scale=1))
            rating = max(1, min(rating, 5))
            self.update_consumption(user_id, item_id, rating)
            return rating
        else:
            return None

    def select_item(self, user_id):
        
        print("Items rated by user:", self.ratings[self.ratings['user_id'] == user_id]['item_id'].tolist())

        unrated_items = items[~items['item_id'].isin(self.ratings[self.ratings['user_id'] == user_id]['item_id'])]
        print("Unrated items:", unrated_items)

        if unrated_items.empty:
            print(f"[WARN] User {user_id} has rated all items. No more to recommend.")
            return None

        unrated_items = unrated_items.copy()
        unrated_items.loc[:, 'predicted_rating'] = unrated_items['item_id'].apply(lambda x: self.predict_rating(user_id, x))

        if unrated_items['predicted_rating'].isnull().all():
            print(f"[WARN] No valid prediction for user {user_id}.")
            return None

        selected_item = unrated_items.loc[unrated_items['predicted_rating'].idxmax()]
        return selected_item['item_id']


    def print_populations(self):
        rounded_item_population = {key: round(value, 2) for key, value in Recommender.item_population.items()}
        rounded_user_population = {key: round(value, 2) for key, value in Recommender.user_population.items()}

        print("Item Population:", rounded_item_population)
        print("User Population:", rounded_user_population)
