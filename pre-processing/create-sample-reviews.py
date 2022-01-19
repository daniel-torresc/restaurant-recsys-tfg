import datetime
import time

import pandas as pd

if __name__ == "__main__":
    # Load business, user and reviews Yelp datasets
    start_time_importing = time.process_time()
    df_business = pd.read_json("dataset/yelp_academic_dataset_business.json", lines=True)
    df_reviews = pd.read_json("dataset/yelp_academic_dataset_review.json", lines=True)
    elapsed_time = datetime.timedelta(seconds=round(time.process_time() - start_time_importing))
    print(f"Elapsed time importing datasets --> {elapsed_time}\n")

    # Retain only needed columns from datasets
    start_time_importing = time.process_time()
    df_business = df_business[['business_id', 'review_count', 'categories']]
    df_reviews = df_reviews[['review_id', 'user_id', 'business_id', 'stars', 'text']]
    elapsed_time = datetime.timedelta(seconds=round(time.process_time() - start_time_importing))
    print(f"Elapsed time keeping specific columns --> {elapsed_time}\n")

    start_time_importing = time.process_time()
    # Filter every business that contains the category 'Restaurants' and has more than 50 reviews
    df_business = df_business.loc[
        (df_business['categories'].str.contains("Restaurants", na=False))
    ]

    # Filter the reviews whose business_id and user_id are in their dfs
    df_reviews = df_reviews.loc[
        (df_reviews['business_id'].isin(df_business['business_id'].tolist()))
    ]

    # Filter minimum reviews
    min_reviews = 30
    while df_reviews['user_id'].value_counts().min() < min_reviews or df_reviews['business_id'].value_counts().min() < min_reviews:
        df_reviews = df_reviews.groupby('business_id').filter(lambda x: len(x) >= min_reviews)
        df_reviews = df_reviews.groupby('user_id').filter(lambda x: len(x) >= min_reviews)

    # Filter max reviews for memory purposes
    max_reviews = 100
    df_reviews = df_reviews.groupby('business_id').filter(lambda x: len(x) <= max_reviews)
    df_reviews = df_reviews.groupby('user_id').filter(lambda x: len(x) <= max_reviews)

    elapsed_time = datetime.timedelta(seconds=round(time.process_time() - start_time_importing))
    print(f"Elapsed time filtering --> {elapsed_time}\n")

    print(f"Number of total users: {len(df_reviews['user_id'].unique())}")
    print(f"Number of total restaurants: {len(df_reviews['business_id'].unique())}\n")

    start_time_importing = time.process_time()
    # Save dataset with only restaurant businesses
    df_reviews.to_json("../processing/dataset/dataset_review_restaurants.json", orient='records', lines=True)

    # Create two samples for testing future developments
    sample = df_reviews.sample(5000, random_state=1)
    sample.to_json("../processing/dataset/dataset_review_restaurants_5k.json", orient='records', lines=True)

    sample = df_reviews.sample(10000, random_state=1)
    sample.to_json("../processing/dataset/dataset_review_restaurants_10k.json", orient='records', lines=True)

    elapsed_time = datetime.timedelta(seconds=round(time.process_time() - start_time_importing))
    print(f"Elapsed time persisting datasets --> {elapsed_time}\n")
