import itertools
import pandas as pd
from recommenders import CosineRecommender, UserKNNRecommender, RestaurantKNNRecommender
from similarity import CosineUserSimilarityAspects, CosineRestaurantSimilarityAspects, CosineUserSimilarityRatings, \
    CosineRestaurantSimilarityRatings
from ratings import Ratings

if __name__ == "__main__":
    df_annotations = pd.read_json("../dataset/annotations_dataset.json", lines=True)

    ratings = Ratings(df_annotations)

    # cb
    recommender = CosineRecommender(ratings=ratings)
    recommendation = recommender.recommend(topn=10)
    print("CosineRecommender - (cb)")
    for user in itertools.islice(recommendation, 4):
        print(f"User [{user}] ->\n{recommendation[user]}")

    # cbub
    similarity_function = CosineUserSimilarityAspects(ratings)
    recommender = UserKNNRecommender(ratings=ratings, sim=similarity_function, k=4)
    recommendation = recommender.recommend(topn=10)
    print("UserKNNRecommender - (cbub)")
    for user in itertools.islice(recommendation, 4):
        print(f"User [{user}] ->\n{recommendation[user]}")

    # cbib
    similarity_function = CosineRestaurantSimilarityAspects(ratings)
    recommender = RestaurantKNNRecommender(ratings=ratings, sim=similarity_function, k=4)
    recommendation = recommender.recommend(topn=10)
    print("RestaurantKNNRecommender - (cbib)")
    for restaurant in itertools.islice(recommendation, 4):
        print(f"Restaurant [{restaurant}] ->\n{recommendation[restaurant]}")

    # ub
    similarity_function = CosineUserSimilarityRatings(ratings)
    recommender = UserKNNRecommender(ratings=ratings, sim=similarity_function, k=4)
    recommendation = recommender.recommend(topn=10)
    print("UserKNNRecommender - (ub)")
    for user in itertools.islice(recommendation, 4):
        print(f"User [{user}] ->\n{recommendation[user]}")

    # ib
    similarity_function = CosineRestaurantSimilarityRatings(ratings)
    recommender = RestaurantKNNRecommender(ratings=ratings, sim=similarity_function, k=4)
    recommendation = recommender.recommend(topn=1)
    print("RestaurantKNNRecommender - (ib)")
    for restaurant in itertools.islice(recommendation, 4):
        print(f"Restaurant [{restaurant}] ->\n{recommendation[restaurant]}")
