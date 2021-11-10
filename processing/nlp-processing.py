import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from multiprocessing import Pool, current_process


def iterate_reviews(df):
    df_aspects = pd.read_csv("../dataset/aspects_restaurants.csv", header=None, names=['key', 'value'])

    # Convert 'aspects' dataframe into dictionary
    aspects_dict = {}
    for _, row in df_aspects.iterrows():
        aspects_dict[row['value']] = row['key']

    # Instantiate SentimentIntensityAnalyzer in order to use VADER (pre-trained sentiment analyzer from nltk)
    sia = SentimentIntensityAnalyzer()

    # Iterate over each review
    row_list = []
    for index, row in df.iterrows():
        if (index + 1) % 1000 == 0:
            print(f"{current_process().name} - Processed {(index+1) % len(df):,} out of {len(df):,} reviews")

        review = row['text']

        # Split review in sentences
        sentences = nltk.sent_tokenize(review)

        # Get annotation for each sentence
        for sentence in sentences:
            feeling = 1 if sia.polarity_scores(sentence)['compound'] > 0 else -1

            words = nltk.word_tokenize(sentence)
            words_tagged = nltk.pos_tag(words)  # Tagging words

            nouns = [word for word, tag in words_tagged if
                     tag.startswith('N') and word in aspects_dict]  # Finding nouns

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
                row_list.append(new_record)

    # Create empty annotations dataframe
    columns = ['user_id', 'review_id', 'restaurant_id', 'rate', 'term', 'aspect', 'feeling']
    annotations_df = pd.DataFrame(row_list, columns=columns)

    return annotations_df


def parallelize_dataframe(df, func, n_cores=8):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


if __name__ == "__main__":
    # Import datasets
    df_reviews = pd.read_json("../dataset/yelp_academic_dataset_review_restaurants.json", lines=True)

    final_df = parallelize_dataframe(df_reviews, iterate_reviews)

    # Dump annotations into json file
    final_df.to_json("../dataset/annotations_dataset.json", orient='records', lines=True)
