import datetime
import random
import time
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from recommender import *
from similarity import *
from ratings import Ratings
from metrics import *


def test_recommenders(recommender: str, ratings: Ratings, metrics: list[Metric], k: int, topn: int, test_items: int):
    start_time = time.process_time()

    if recommender == "cb":
        if test_items:
            all_users = list(ratings.users())
            random.Random(2).shuffle(all_users)
            test_users = all_users[:test_items]
            test_recommender(CosineRecommender(ratings), topn, metrics, test_users)
        else:
            test_recommender(CosineRecommender(ratings), topn, metrics)
    elif recommender == "ub":
        if test_items:
            all_users = list(ratings.users())
            random.Random(2).shuffle(all_users)
            test_users = all_users[:test_items]
            test_recommender(UserKNNRecommender(ratings, CosineUserSimilarityRatings(ratings), k), topn, metrics, test_users)
        else:
            test_recommender(UserKNNRecommender(ratings, CosineUserSimilarityRatings(ratings), k), topn, metrics)
    elif recommender == "cbub":
        if test_items:
            all_users = list(ratings.users())
            random.Random(2).shuffle(all_users)
            test_users = all_users[:test_items]
            test_recommender(UserKNNRecommender(ratings, CosineUserSimilarityAspects(ratings), k), topn, metrics, test_users)
        else:
            test_recommender(UserKNNRecommender(ratings, CosineUserSimilarityAspects(ratings), k), topn, metrics)
    elif recommender == "ib":
        if test_items:
            all_restaurants = list(ratings.restaurants())
            random.Random(2).shuffle(all_restaurants)
            test_restaurants = all_restaurants[:test_items]
            test_recommender(RestaurantKNNRecommender(ratings, CosineRestaurantSimilarityRatings(ratings), k), topn, metrics, test_restaurants)
        else:
            test_recommender(RestaurantKNNRecommender(ratings, CosineRestaurantSimilarityRatings(ratings), k), topn, metrics)
    elif recommender == "cbib":
        if test_items:
            all_restaurants = list(ratings.restaurants())
            random.Random(2).shuffle(all_restaurants)
            test_restaurants = all_restaurants[:test_items]
            test_recommender(RestaurantKNNRecommender(ratings, CosineRestaurantSimilarityAspects(ratings), k), topn, metrics, test_restaurants)
        else:
            test_recommender(RestaurantKNNRecommender(ratings, CosineRestaurantSimilarityAspects(ratings), k), topn, metrics)

    elapsed_time = datetime.timedelta(seconds=round(time.process_time() - start_time))

    print(f"\nElapsed time testing {recommender} recommender --> {elapsed_time}")


def test_recommender(recommender: Recommender, topn: int, metrics: list[Metric], test_items: list = None):
    print(f"\nTesting {recommender} - Top {topn}")

    print("\tBuilding recommendations...", end='', flush=True)
    recommendation = recommender.recommend(topn, test_items)
    print("DONE")

    for metric in metrics:
        mean, stddev = metric.compute(recommendation)
        print(f"\t{metric}\n\t\tmean={mean}\tstddev={stddev}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ts", "--test_size",
        type=float,
        choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        default=0.2,
        required=False,
        help="Test size percentage - [0-1)",
    )
    parser.add_argument(
        "-r", "--recommender",
        type=str,
        choices=["cb", "ub", "cbub", "ib", "cbib"],
        required=True,
        help="Recommender to use")
    parser.add_argument(
        "-ti", "--test_items",
        type=int,
        default=None,
        required=False,
        help="Number of items to recommend")
    parser.add_argument(
        "-th", "--threshold",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        required=False,
        help="Threshold for testing metrics")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        choices=["5k", "10k", "complete"],
        required=True,
        help="Dataset to test")
    parser.add_argument(
        "-k",
        type=int,
        required=True,
        help="Number of K neighbors users for the recommender")
    parser.add_argument(
        "--topn",
        type=int,
        required=True,
        help="Size of the recommendation ranking")
    args = parser.parse_args()

    start_time_importing = time.process_time()
    if args.dataset == "5k":
        df_annotations = pd.read_pickle("dataset/annotations_dataset_5k.pickle")
    elif args.dataset == "10k":
        df_annotations = pd.read_pickle("dataset/annotations_dataset_10k.pickle")
    else:
        df_annotations = pd.read_pickle("dataset/annotations_dataset.pickle")
    elapsed_time_importing = datetime.timedelta(seconds=round(time.process_time() - start_time_importing))
    print(f"Elapsed time importing dataset --> {elapsed_time_importing}")

    start_time_importing = time.process_time()
    train_df, test_df = train_test_split(df_annotations, test_size=args.test_size, random_state=1)
    elapsed_time_importing = datetime.timedelta(seconds=round(time.process_time() - start_time_importing))
    print(f"Elapsed time spliting dataset into test-train --> {elapsed_time_importing}")

    start_time_importing = time.process_time()
    train = Ratings(train_df)
    test = Ratings(test_df)
    elapsed_time_importing = datetime.timedelta(seconds=round(time.process_time() - start_time_importing))
    print(f"Elapsed time generating ratings --> {elapsed_time_importing}")

    start_time_importing = time.process_time()
    metrics = [
        Precision(test, cutoff=5, threshold=args.threshold),
        Recall(test, cutoff=5, threshold=args.threshold),
        Precision(test, cutoff=10, threshold=args.threshold),
        Recall(test, cutoff=10, threshold=args.threshold),
        Precision(test, cutoff=20, threshold=args.threshold),
        Recall(test, cutoff=20, threshold=args.threshold),
        Precision(test, cutoff=50, threshold=args.threshold),
        Recall(test, cutoff=50, threshold=args.threshold)
    ]
    elapsed_time_importing = datetime.timedelta(seconds=round(time.process_time() - start_time_importing))
    print(f"Elapsed time loading metrics --> {elapsed_time_importing}\n")

    test_recommenders(args.recommender, train, metrics, k=args.k, topn=args.topn, test_items=args.test_items)
