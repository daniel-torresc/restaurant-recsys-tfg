

class Ratings:

    def __init__(self, df):
        self.all_restaurants = sorted(df['restaurant_id'].unique())  # list of all restaurants
        self.all_users = sorted(df['user_id'].unique())  # list of all users

        self.user_ratings = {}  # weighted aspects for each user
        self.restaurant_ratings = {}  # weighted aspects for each restaurant

        # Calculate the weight of each aspect for every user
        for user in self.all_users:
            aux_df = df.loc[df['user_id'] == user]  # Create subset with only the rows where user == user_id
            self.user_ratings[user] = {}

            for _, row in aux_df.iterrows():
                aspect = row['aspect']

                # TODO: hacer una buena ponderacion entre el rate y el feeling
                weighted_score = 0.6 * row['rate'] + 0.4 * row['feeling'] * 5

                self.user_ratings[user].setdefault(aspect, weighted_score)
                self.user_ratings[user][aspect] = 0.5 * self.user_ratings[user][aspect] + 0.5 * weighted_score

        # Calculate the weight of each aspect for every restaurant
        for restaurant in self.all_restaurants:
            aux_df = df.loc[df['restaurant_id'] == restaurant]

            self.restaurant_ratings[restaurant] = {}
            for _, row in aux_df.iterrows():
                aspect = row['aspect']

                # TODO: hacer una buena ponderacion entre el rate y el feeling
                weighted_score = 0.6 * row['rate'] + 0.4 * row['feeling'] * 5

                self.restaurant_ratings[restaurant].setdefault(aspect, weighted_score)
                self.restaurant_ratings[restaurant][aspect] = 0.5 * self.restaurant_ratings[restaurant][aspect] + 0.5 * weighted_score

    def rating(self, item, aspect):
        if item in self.users():
            return self.user_ratings[item][aspect]
        elif item in self.restaurants():
            return self.restaurant_ratings[item][aspect]
        else:
            return None

    def user_aspects(self, user):
        return self.user_ratings[user] if user in self.user_ratings else None

    def restaurant_aspects(self, restaurant):
        return self.restaurant_ratings[restaurant] if restaurant in self.restaurant_ratings else None

    def users(self):
        return self.all_users

    def restaurants(self):
        return self.all_restaurants
