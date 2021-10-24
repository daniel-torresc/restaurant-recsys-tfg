import itertools
import pandas as pd
from recommenders import CosineRecommender
from ratings import Ratings

if __name__ == "__main__":
    df_annotations = pd.read_json("../dataset/annotations_dataset.json", lines=True)

    ratings = Ratings(df_annotations)
    recommender = CosineRecommender(ratings)

    recommendation = recommender.recommend(topn=10)

    for user in itertools.islice(recommendation, 4):
        print(f"User [{user}] ->\n{recommendation[user]}")
