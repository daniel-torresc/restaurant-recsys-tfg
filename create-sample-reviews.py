import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("dataset/yelp_academic_dataset_review.json", lines=True)
    df = df[['review_id', 'user_id', 'business_id', 'stars', 'text']]
    sample = df.sample(10000)
    sample.to_json("dataset/yelp_academic_dataset_review_sample.json", orient='records', lines=True)
