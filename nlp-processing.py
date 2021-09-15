import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

if __name__ == "__main__":
    # Import datasets
    df_reviews = pd.read_json("dataset/yelp_academic_dataset_review_1k.json", lines=True)
    df_aspects = pd.read_csv("dataset/aspects_restaurants.csv", header=None, names=['key', 'value'])
    df_lexicon = pd.read_csv("dataset/lexicon_restaurants.csv", header=None, names=['aspect', 'lexicon', 'weight'])
    df_modifiers = pd.read_csv("dataset/adjective-modifiers.csv", header=None, names=['key', 'value'])

    # Create empty annotations dataframe
    columns = ['user_id', 'review_id', 'rate', 'term', 'aspect', 'lexicon', 'modifier', 'is_negated', 'lexicon_weight', 'modifier_weight']
    annotations_df = pd.DataFrame(columns=columns)

    # Convert 'aspects' dataframe into dictionary
    aspects_dict = {}
    for _, row in df_aspects.iterrows():
        aspects_dict.setdefault(row['key'], [])
        aspects_dict[row['key']].append(row['value'])

    # Convert 'lexicon' dataframe into dictionary
    lexicon_dict = {}
    for _, row in df_lexicon.iterrows():
        lexicon_dict.setdefault(row['aspect'], {})
        lexicon_dict[row['aspect']][row['lexicon']] = row['weight']

    # Convert 'modifiers' dataframe into dictionary
    modifiers_dict = {}
    for _, row in df_modifiers.iterrows():
        modifiers_dict[row['key']] = row['value']

    # Instantiate SentimentIntensityAnalyzer in order to use VADER (pre-trained sentiment analyzer form nltk)
    sia = SentimentIntensityAnalyzer()

    for index, row in df_reviews.iterrows():
        print(f"{index+1} out of {len(df_reviews)}")

        user_id = row['user_id']
        review_id = row['review_id']
        rate = row['stars']
        review = row['text']

        # Split review in sentences
        sentences = nltk.sent_tokenize(review)

        # Split each sentence into words
        for sentence in sentences:
            text = nltk.word_tokenize(sentence)
            text_tagged = nltk.pos_tag(text)

            is_negated = 1 if sia.polarity_scores(sentence)['compound'] <= 0 else -1

            nouns = [n[0] for n in text_tagged if n[1].startswith('N')]  # Finding nouns
            adjectives = [n[0] for n in text_tagged if n[1].startswith('J')]  # Finding adjectives
            modifiers = [n[0] for n in text_tagged if n[1].startswith('RB')]  # Finding adverbs modifiers

            # nouns = list(filter(lambda x: x[1].startswith('N'), text_tagged))  # Finding nouns
            # adjectives = list(filter(lambda x: x[1].startswith('J'), text_tagged))  # Finding adjectives
            # modifiers = list(filter(lambda x: x[1].startswith('RB'), text_tagged))  # Finding adverbs modifiers

            for token, _ in nouns:  # Finding nouns
                for aspect in aspects_dict:
                    if token in aspects_dict[aspect]:  # Checking if the noun correspond to one of our aspects
                        term = token

                        # TODO: extract lexicon and modifier from sentence
                        lexicon = None
                        modifier = None
                        lexicon_weight = None
                        modifier_weight = None

                        new_record = [user_id, review_id, rate, term, aspect, lexicon, modifier, is_negated, lexicon_weight, modifier_weight]
                        annotations_df.loc[len(annotations_df)] = new_record

    # Dump annotations df into persisting file
    annotations_df.to_pickle("dataset/annotations_dataset.pkl")
