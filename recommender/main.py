import datetime
import itertools
import random
import time

import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from recommenders import CosineRecommender, UserKNNRecommender, RestaurantKNNRecommender
from similarity import CosineUserSimilarityAspects, CosineRestaurantSimilarityAspects, CosineUserSimilarityRatings, \
    CosineRestaurantSimilarityRatings
from ratings import Ratings


def test_recommenders(recommender, ratings, k, topn, test_items=None):

    start_time = time.process_time()
    if recommender == "cb":
        if test_items:
            all_users = list(ratings.users())
            random.Random(2).shuffle(all_users)
            test_users = all_users[:test_items]
            test_recommender(CosineRecommender(ratings), topn, test_users)
        else:
            test_recommender(CosineRecommender(ratings), topn)
    elif recommender == "ub":
        if test_items:
            all_users = list(ratings.users())
            random.Random(2).shuffle(all_users)
            test_users = all_users[:test_items]
            test_recommender(UserKNNRecommender(ratings, CosineUserSimilarityRatings(ratings), k), topn, test_users)
        else:
            test_recommender(UserKNNRecommender(ratings, CosineUserSimilarityRatings(ratings), k), topn)
    elif recommender == "cbub":
        if test_items:
            all_users = list(ratings.users())
            random.Random(2).shuffle(all_users)
            test_users = all_users[:test_items]
            test_recommender(UserKNNRecommender(ratings, CosineUserSimilarityAspects(ratings), k), topn, test_users)
        else:
            test_recommender(UserKNNRecommender(ratings, CosineUserSimilarityAspects(ratings), k), topn)
    elif recommender == "ib":
        if test_items:
            all_restaurants = list(ratings.restaurants())
            random.Random(2).shuffle(all_restaurants)
            test_restaurants = all_restaurants[:test_items]
            test_recommender(RestaurantKNNRecommender(ratings, CosineRestaurantSimilarityRatings(ratings), k), topn, test_restaurants)
        else:
            test_recommender(RestaurantKNNRecommender(ratings, CosineRestaurantSimilarityRatings(ratings), k), topn)
    elif recommender == "cbib":
        if test_items:
            all_restaurants = list(ratings.restaurants())
            random.Random(2).shuffle(all_restaurants)
            test_restaurants = all_restaurants[:test_items]
            test_recommender(RestaurantKNNRecommender(ratings, CosineRestaurantSimilarityAspects(ratings), k), topn, test_restaurants)
        else:
            test_recommender(RestaurantKNNRecommender(ratings, CosineRestaurantSimilarityAspects(ratings), k), topn)
    elapsed_time = datetime.timedelta(seconds=round(time.process_time() - start_time))

    print(f"\nElapsed time testing recommender --> {elapsed_time}")


def test_recommender(recommender, topn, test_users=None):
    print(f"Testing {recommender} - Top {topn}")

    recommendation = recommender.recommend(topn, test_users)
    for item in itertools.islice(recommendation, 10):
        if "user" in str(recommender).lower() or "cosine" in str(recommender).lower():
            print(f"User [{item}] ->\n{recommendation[item]}")
        else:
            print(f"Restaurant [{item}] ->\n{recommendation[item]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ts", "--test_size",
        type=float,
        default=0.2,
        required=False,
        help="Test size percentage - [0-1)",
    )
    parser.add_argument(
        "-r", "--recommender",
        choices=["cb", "ub", "cbub", "ib", "cbib"],
        required=True,
        help="Recommender to use")
    parser.add_argument(
        "-ti", "--test_items",
        type=int,
        default=None,
        required=False,
        help="Number of items to recommend")
    args = parser.parse_args()

    start_time = time.process_time()
    df_annotations = pd.read_pickle("dataset/annotations_dataset_5k.pickle")
    elapsed_time = datetime.timedelta(seconds=round(time.process_time() - start_time))
    print(f"\nElapsed time importing dataset --> {elapsed_time}\n")

    train_df, test_df = train_test_split(df_annotations, test_size=args.test_size, random_state=1)
    train = Ratings(train_df)
    test = Ratings(test_df)

    test_recommenders(args.recommender, train, k=10, topn=20, test_items=args.test_items)
