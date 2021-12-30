import itertools
import pandas as pd
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
    df_annotations = pd.read_pickle("dataset/annotations_dataset.pickle")

    test_recommenders(Ratings(df_annotations), k=10, topn=5)
