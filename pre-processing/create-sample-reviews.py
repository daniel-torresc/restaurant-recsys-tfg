import pandas as pd

if __name__ == "__main__":
    # Load both business and reviews datasets
    df_business = pd.read_json("../dataset/yelp_academic_dataset_business.json", lines=True)
    df_reviews = pd.read_json("../dataset/yelp_academic_dataset_review.json", lines=True)

    # Filter every business that contains the category Restaurants
    df_business = df_business.loc[df_business['categories'].str.contains("Restaurants", na=False)]

    # Filter the reviews whose business_id is a restaurant
    df_reviews = df_reviews.loc[df_reviews['business_id'].isin(df_business['business_id'].tolist())]

    # Remove unwanted columns from reviews dataset
    df_reviews = df_reviews[['review_id', 'user_id', 'business_id', 'stars', 'text']]

    # Save dataset with only restaurant businesses
    df_reviews.to_json("../dataset/yelp_academic_dataset_review_restaurants.json", orient='records', lines=True)

    # Save dataset samples
    sample = df_reviews.sample(1000)
    sample.to_json("../dataset/yelp_academic_dataset_review_1k.json", orient='records', lines=True)

    sample = df_reviews.sample(5000)
    sample.to_json("../dataset/yelp_academic_dataset_review_5k.json", orient='records', lines=True)

    sample = df_reviews.sample(10000)
    sample.to_json("../dataset/yelp_academic_dataset_review_10k.json", orient='records', lines=True)
