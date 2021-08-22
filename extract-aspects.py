import pandas as pd
from spellchecker import SpellChecker

# Import datasets
df_reviews = pd.read_json("dataset/yelp_academic_dataset_review_sample.json", lines=True)
df_reviews['aspects'] = None
df_aspects = pd.read_csv("dataset/aspects_restaurants.csv", header=None, names=['key', 'value'])

spell = SpellChecker(language='en')
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

    # Correct misspelled words
    review_words = review.lower().split()
    corrected_words = []
    for word in review_words:
        word = spell.correction(word)
        corrected_words.append(word)

    # Add the review's words to aspects if needed
    aspects = []
    for aspect in aspects_dict:
        for a in aspects_dict[aspect]:
            if a in corrected_words:
                aspects.append(aspect)

    aspects = list(set(aspects))
    df_reviews.loc[index, 'aspects'] = str(aspects)
