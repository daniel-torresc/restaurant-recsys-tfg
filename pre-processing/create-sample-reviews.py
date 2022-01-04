import pandas as pd

if __name__ == "__main__":
    # Load business, user and reviews Yelp datasets
    df_business = pd.read_json("dataset/yelp_academic_dataset_business.json", lines=True)
    df_user = pd.read_json("dataset/yelp_academic_dataset_user.json", lines=True)
    df_reviews = pd.read_json("dataset/yelp_academic_dataset_review.json", lines=True)

    # Retain only needed columns from datasets
    df_business = df_business[['business_id', 'review_count', 'categories']]
    df_user = df_user[['user_id', 'review_count']]
    df_reviews = df_reviews[['review_id', 'user_id', 'business_id', 'stars', 'text']]

    # Filter every business that contains the category 'Restaurants' and has more than 50 reviews
    perc_99 = int(df_business['review_count'].quantile(0.99))
    df_business = df_business.loc[
        (df_business['categories'].str.contains("Restaurants", na=False)) &
        (df_business['review_count'] >= 10) &
        (df_business['review_count'] <= perc_99)
    ]

    # Filter every user with more than 50 reviews
    perc_99 = int(df_user['review_count'].quantile(0.99))
    df_user = df_user.loc[
        (df_user['review_count'] >= 10) &
        (df_user['review_count'] <= perc_99)
    ]

    # Filter the reviews whose business_id and user_id are in their dfs
    df_reviews = df_reviews.loc[
        (df_reviews['business_id'].isin(df_business['business_id'].tolist())) &
        (df_reviews['user_id'].isin(df_user['user_id'].tolist()))
    ]

    # Save dataset with only restaurant businesses
    df_reviews.to_json("../processing/dataset/dataset_review_restaurants.json", orient='records', lines=True)

    # Create two samples for testing future developments
    sample = df_reviews.sample(5000, random_state=1)
    sample.to_json("../processing/dataset/dataset_review_restaurants_5k.json", orient='records', lines=True)

    sample = df_reviews.sample(10000, random_state=1)
    sample.to_json("../processing/dataset/dataset_review_restaurants_10k.json", orient='records', lines=True)
