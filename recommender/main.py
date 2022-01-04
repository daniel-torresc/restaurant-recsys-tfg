import itertools
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from recommenders import CosineRecommender, UserKNNRecommender, RestaurantKNNRecommender
from similarity import CosineUserSimilarityAspects, CosineRestaurantSimilarityAspects, CosineUserSimilarityRatings, \
    CosineRestaurantSimilarityRatings
from ratings import Ratings


def test_recommenders(ratings, k, topn):
    test_recommender(CosineRecommender(ratings), topn)

    similarities = [
        CosineUserSimilarityRatings(ratings),
        CosineUserSimilarityAspects(ratings),
        CosineRestaurantSimilarityRatings(ratings),
        CosineRestaurantSimilarityAspects(ratings)
    ]

    recommenders = []
    for sim in similarities:
        recommenders.append(UserKNNRecommender(ratings, sim, k))
        recommenders.append(RestaurantKNNRecommender(ratings, sim, k))

    for recommender in recommenders:
        test_recommender(recommender, topn)


def test_recommender(recommender, topn):
    print(f"Testing {recommender} - Top {topn}")

    recommendation = recommender.recommend(topn)
    if "user" in str(recommender).lower():
        for item in itertools.islice(recommendation, 4):
            print(f"User [{item}] ->\n{recommendation[item]}")
    else:
        for item in itertools.islice(recommendation, 4):
            print(f"Restaurant [{item}] ->\n{recommendation[item]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--size",
        default=0.2,
        type=float,
        choices=range(1),
        help="Test size percentage - [0-1)",
    )

    args = parser.parse_args()

    df_annotations = pd.read_pickle("dataset/annotations_dataset_5k.pickle")

    train_df, test_df = train_test_split(df_annotations, test_size=args.size, random_state=1)
    train = Ratings(train_df)
    test = Ratings(test_df)

    test_recommenders(train, k=10, topn=5)
