import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def get_associated_lexicon(term: str, aspect: str, adjectives: list, words: list, lexicon_dict: dict):
    # TODO: check when the term is more than once in words
    # num_occurrences = words.count(term)
    # if num_occurrences == 1:

    term_index = words.index(term)

    for adjective in adjectives:
        adjective_index = words.index(adjective)
        # TODO: check when the adjective is more than once in words

        if term_index - 1 == adjective_index or term_index + 2 == adjective_index:  # TODO: check after the verb, when the sentence is negated
            if adjective in lexicon_dict[aspect]:
                return adjective, lexicon_dict[aspect][adjective]

    return None, None


def get_associated_modifier(lexicon: str, modifiers: list, words: list, modifiers_dict: dict):
    lexicon_index = words.index(lexicon)

    for modifier in modifiers:
        modifier_index = words.index(modifier)

        if lexicon_index - 1 == modifier_index:
            if lexicon in modifiers_dict:
                return modifier, modifiers_dict[lexicon]

    return None, None


if __name__ == "__main__":
    # Import datasets
    df_reviews = pd.read_json("dataset/yelp_academic_dataset_review_1k.json", lines=True)
    df_aspects = pd.read_csv("dataset/aspects_restaurants.csv", header=None, names=['key', 'value'])
    df_lexicon = pd.read_csv("dataset/lexicon_restaurants.csv", header=None, names=['aspect', 'lexicon', 'weight'])
    df_modifiers = pd.read_csv("dataset/adjective-modifiers.csv", header=None, names=['modifier', 'weight'])

    # Create empty annotations dataframe
    columns = ['user_id', 'review_id', 'rate', 'term', 'aspect', 'lexicon', 'modifier', 'feeling', 'lexicon_weight', 'modifier_weight']
    annotations_df = pd.DataFrame(columns=columns)
    # dtype = {
    #     'user_id': str,
    #     'review_id': str,
    #     'rate': int,
    #     'term': str,
    #     'aspect': str,
    #     'lexicon': str,
    #     'modifier': str,
    #     'feel': bool,
    #     'lexicon_weight': int,
    #     'modifier_weight': int
    # }
    # annotations_df.astype(dtype)

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
        modifiers_dict[row['modifier']] = row['weight']

    # Instantiate SentimentIntensityAnalyzer in order to use VADER (pre-trained sentiment analyzer from nltk)
    sia = SentimentIntensityAnalyzer()

    # Iterate over each review
    for index, row in df_reviews.iterrows():
        if (index + 1) % 10 == 0:
            print(f"{index + 1} out of {len(df_reviews)}")

        user_id = row['user_id']
        review_id = row['review_id']
        rate = row['stars']
        review = row['text']

        # Split review in sentences
        sentences = nltk.sent_tokenize(review)

        # Get annotation for each sentence
        for sentence in sentences:
            feeling = 1 if sia.polarity_scores(sentence)['compound'] > 0 else -1

            words = nltk.word_tokenize(sentence)
            words_tagged = nltk.pos_tag(words)  # Tagging words

            nouns = [word for word, tag in words_tagged if tag.startswith('N')]  # Finding nouns
            adjectives = [word for word, tag in words_tagged if tag.startswith('J')]  # Finding adjectives
            modifiers = [word for word, tag in words_tagged if tag.startswith('RB')]  # Finding adverbs (modifiers)

            for noun in nouns:  # Finding nouns
                for aspect in aspects_dict:
                    if noun in aspects_dict[aspect]:  # Checking if the noun correspond to one of our aspects
                        term = noun

                        lexicon, lexicon_weight = get_associated_lexicon(term, aspect, adjectives, words, lexicon_dict)
                        modifier, modifier_weight = get_associated_modifier(lexicon, modifiers, words, modifiers_dict) if lexicon is not None else None, None

                        new_record = [user_id, review_id, rate, term, aspect, lexicon, modifier, feeling,
                                      lexicon_weight, modifier_weight]
                        annotations_df.loc[len(annotations_df)] = new_record

    # Dump annotations df into persisting file
    # annotations_df.to_pickle("dataset/annotations_dataset.pkl")

    # Dump annotations into json file
    annotations_df.to_json("dataset/annotations_dataset.json", orient='records', lines=True)
