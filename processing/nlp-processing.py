import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


if __name__ == "__main__":
    # Import datasets
    df_reviews = pd.read_json("../dataset/yelp_academic_dataset_review_restaurants.json", lines=True)
    df_aspects = pd.read_csv("../dataset/aspects_restaurants.csv", header=None, names=['key', 'value'])

    # Create empty annotations dataframe
    columns = ['user_id', 'review_id', 'restaurant_id', 'rate', 'term', 'aspect', 'feeling']
    annotations_df = pd.DataFrame(columns=columns)

    # Convert 'aspects' dataframe into dictionary
    aspects_dict = {}
    for _, row in df_aspects.iterrows():
        aspects_dict[row['value']] = row['key']

    # Instantiate SentimentIntensityAnalyzer in order to use VADER (pre-trained sentiment analyzer from nltk)
    sia = SentimentIntensityAnalyzer()

    # Iterate over each review
    for index, row in df_reviews.iterrows():
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1:,} out of {len(df_reviews):,} reviews")

        review = row['text']

        # Split review in sentences
        sentences = nltk.sent_tokenize(review)

        # Get annotation for each sentence
        for sentence in sentences:
            feeling = 1 if sia.polarity_scores(sentence)['compound'] > 0 else -1

            words = nltk.word_tokenize(sentence)
            words_tagged = nltk.pos_tag(words)  # Tagging words

            nouns = [word for word, tag in words_tagged if tag.startswith('N') and word in aspects_dict]  # Finding nouns

            for term in nouns:
                new_record = [
                    row['user_id'],
                    row['review_id'],
                    row['business_id'],
                    row['stars'],
                    term,
                    aspects_dict[term],
                    feeling
                ]
                annotations_df.loc[len(annotations_df)] = new_record

    # Dump annotations into json file
    annotations_df.to_json("../dataset/annotations_dataset.json", orient='records', lines=True)
