import itertools
import pandas as pd
from recommenders import CosineRecommender, UserKNNRecommender, RestaurantKNNRecommender
from similarity import CosineUserSimilarity, CosineRestaurantSimilarity
from ratings import Ratings

if __name__ == "__main__":
    df_annotations = pd.read_json("../dataset/annotations_dataset.json", lines=True)

    ratings = Ratings(df_annotations)

    # CosineRecommender
    # recommender = CosineRecommender(ratings=ratings)
    # recommendation = recommender.recommend(topn=10)
    # print("CosineRecommender - (cb)")
    # for user in itertools.islice(recommendation, 4):
    #     print(f"User [{user}] ->\n{recommendation[user]}")

    # UserKNNRecommender
    # similarity_function = CosineUserSimilarity(ratings)
    # recommender = UserKNNRecommender(ratings=ratings, sim=similarity_function, k=4)
    # recommendation = recommender.recommend(topn=10)
    # print("UserKNNRecommender - (cbub)")
    # for user in itertools.islice(recommendation, 4):
    #     print(f"User [{user}] ->\n{recommendation[user]}")

    # RestaurantKNNRecommender
    similarity_function = CosineRestaurantSimilarity(ratings)
    recommender = RestaurantKNNRecommender(ratings=ratings, sim=similarity_function, k=4)
    recommendation = recommender.recommend(topn=10)
    print("RestaurantKNNRecommender - (cbib)")
    for restaurant in itertools.islice(recommendation, 4):
        print(f"Restaurant [{restaurant}] ->\n{recommendation[restaurant]}")
