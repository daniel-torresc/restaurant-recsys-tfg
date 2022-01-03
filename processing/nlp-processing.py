import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from multiprocessing import Pool, current_process, cpu_count


def iterate_reviews(df):
    df_aspects = pd.read_csv("dataset/aspects_restaurants.csv", header=None, names=['key', 'value'])

    # Convert 'aspects' dataframe into dictionary
    aspects_dict = {row['value']: row['key'] for _, row in df_aspects.iterrows()}
    # Instantiate SentimentIntensityAnalyzer in order to use VADER (pre-trained sentiment analyzer from nltk)
    sia = SentimentIntensityAnalyzer()

    # Iterate over each review
    row_list = []
    for index, row in df.iterrows():
        if (index + 1) % 1000 == 0:
            print(f"{current_process().name:>20} - Processed {(index+1) % len(df):>8} out of {len(df):>8} reviews")

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

    # Create partial annotations dataframe
    columns = ['user_id', 'review_id', 'restaurant_id', 'rate', 'term', 'aspect', 'feeling']
    return pd.DataFrame(row_list, columns=columns)


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, cpu_count())
    pool = Pool(cpu_count())
    df_concat = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df_concat


if __name__ == "__main__":
    # Import datasets
    df_reviews = pd.read_json("dataset/dataset_review_restaurants.json", lines=True)
    df_reviews.name = "annotations_dataset"

    df_reviews_5k = pd.read_json("dataset/dataset_review_restaurants_5k.json", lines=True)
    df_reviews_5k.name = "annotations_dataset_5k"

    df_reviews_10k = pd.read_json("dataset/dataset_review_restaurants_10k.json", lines=True)
    df_reviews_10k.name = "annotations_dataset_10k"

    for dframe in [df_reviews_5k, df_reviews_10k, df_reviews]:
        final_df = parallelize_dataframe(dframe, iterate_reviews)

        # Dump annotations into file
        final_df.to_pickle(f"../recommender/dataset/{dframe.name}.pickle")
        final_df.to_json(f"../recommender/dataset/{dframe.name}.json", orient='records', lines=True)

        print(f"Finished processing {dframe.name}...\n")
