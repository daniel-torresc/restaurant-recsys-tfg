import pandas as pd


class Ratings:

    def __init__(self, df):
        self.aspects = sorted(df['aspect'].unique())  # list of all aspects in the reviews
        self.restaurants = sorted(df['restaurant_id'].unique())  # list of all restaurants
        self.users = sorted(df['user_id'].unique())  # list of all users

        self.user_aspects = {}  # weighted aspects for each user
        self.restaurant_aspects = {}  # weighted aspects for each restaurant

        # Calculate the weight of each aspect for every user
        for user in self.users:
            aux_df = df.loc[df['user_id'] == user]
            self.user_aspects[user] = {}

            for _, row in aux_df.iterrows():
                aspect = row['aspect']

                # TODO: hacer una buena ponderacion entre el rate y el feeling
                weighted_score = 0.6 * row['rate'] + 0.4 * row['feeling'] * 5

                self.user_aspects[user].setdefault(aspect, weighted_score)
                self.user_aspects[user][aspect] = 0.5*self.user_aspects[user][aspect] + 0.5*weighted_score

        # Calculate the weight of each aspect for every restaurant
        for restaurant in self.restaurants:
            aux_df = df.loc[df['restaurant_id'] == restaurant]

            self.restaurant_aspects[restaurant] = {}
            for _, row in aux_df.iterrows():
                aspect = row['aspect']

                # TODO: hacer una buena ponderacion entre el rate y el feeling
                weighted_score = 0.6 * row['rate'] + 0.4 * row['feeling'] * 5

                self.restaurant_aspects[restaurant].setdefault(aspect, weighted_score)
                self.restaurant_aspects[restaurant][aspect] = 0.5 * self.restaurant_aspects[restaurant][aspect] + 0.5 * weighted_score


if __name__ == "__main__":
    df_annotations = pd.read_json("../dataset/annotations_dataset.json", lines=True)

    ratings = Ratings(df_annotations)

    print("DONE")
