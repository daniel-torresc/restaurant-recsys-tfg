import pandas as pd

if __name__ == "__main__":
    # Import datasets
    df_reviews = pd.read_json("dataset/yelp_academic_dataset_review_sample.json", lines=True)
    df_aspects = pd.read_csv("dataset/aspects_restaurants.csv", header=None, names=['key', 'value'])

    punctuation_marks = ".,!?:;-()[]"

    # Convert 'aspects' dataframe into dictionary
    aspects_dict = {}
    for _, row in df_aspects.iterrows():
        aspects_dict.setdefault(row['key'], [])
        aspects_dict[row['key']].append(row['value'])

    # Filling 'aspects' attribute in 'reviews' dataset
    for index, row in df_reviews.iterrows():
        review = row['text']

        # Remove of punctuation marks in review
        for char in punctuation_marks:
            review = review.replace(char, "")

        # Check which aspects correspond to the review
        review_words = review.lower().split()
        aspects = []
        for aspect in aspects_dict:
            for a in aspects_dict[aspect]:
                if a in review_words:
                    aspects.append(aspect)

        aspects = list(set(aspects))
        df_reviews.loc[index, 'aspects'] = str(aspects)

    # Drop reviews with no aspects from dataset
    df_reviews.drop(df_reviews[df_reviews.aspects == "[]"].index, inplace=True)

    # Dump 'aspects' into json file as new attribute
    df_reviews.to_json("dataset/yelp_academic_dataset_review_sample.json", orient='records', lines=True)
