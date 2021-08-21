import pandas as pd
from spellchecker import SpellChecker

df = pd.read_json("dataset/yelp_academic_dataset_review_sample.json", lines=True)
df['aspects'] = None

spell = SpellChecker(language='en')
punct_marks = ".,!?:;-()[]"

for index, row in df.iterrows():
    review = row['text']

    # Remove of punctuation marks in review
    for char in punct_marks:
        review = review.replace(char, "")

    # Correct misspelled words
    review_words = review.lower().split()
    corrected_words = []
    for word in review_words:
        word = spell.correction(word)
        corrected_words.append(word)

    print(corrected_words)