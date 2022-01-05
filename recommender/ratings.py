

class Ratings:

    def __init__(self, df):
        self.all_users = sorted(df['user_id'].unique())  # list of all users
        self.all_restaurants = sorted(df['restaurant_id'].unique())  # list of all restaurants

        self.user_aspects = {}  # weighted aspects for each user
        self.restaurant_aspects = {}  # weighted aspects for each restaurant

        self.ratings_of_user = {}  # scores of the user to the restaurants
        self.ratings_of_restaurant = {}  # scores of the restaurant from the users

        # Calculate the weight of each aspect for every user
        for user in self.all_users:
            aux_df = df.loc[df['user_id'] == user]  # Create subset with only the rows where user == user_id
            self.user_aspects[user] = {}
            self.ratings_of_user[user] = {}

            for _, row in aux_df.iterrows():
                aspect = row['aspect']
                rate = row['rate']
                restaurant = row['restaurant_id']

                # TODO: hacer una buena ponderacion entre el rate y el feeling
                weighted_score = 0.6 * rate + 0.4 * row['feeling'] * 5

                self.user_aspects[user].setdefault(aspect, weighted_score)
                self.user_aspects[user][aspect] = 0.5 * self.user_aspects[user][aspect] + 0.5 * weighted_score

                self.ratings_of_user[user][restaurant] = rate

        # Calculate the weight of each aspect for every restaurant
        for restaurant in self.all_restaurants:
            aux_df = df.loc[df['restaurant_id'] == restaurant]
            self.restaurant_aspects[restaurant] = {}
            self.ratings_of_restaurant[restaurant] = {}

            for _, row in aux_df.iterrows():
                aspect = row['aspect']
                rate = row['rate']
                user = row['user_id']

                # TODO: hacer una buena ponderacion entre el rate y el feeling
                weighted_score = 0.6 * rate + 0.4 * row['feeling'] * 5

                self.restaurant_aspects[restaurant].setdefault(aspect, weighted_score)
                self.restaurant_aspects[restaurant][aspect] = 0.5 * self.restaurant_aspects[restaurant][aspect] + 0.5 * weighted_score

                self.ratings_of_restaurant[restaurant][user] = rate

    def ratings(self, item):
        if item in self.users():
            return self.ratings_of_user[item]
        elif item in self.restaurants():
            return self.ratings_of_restaurant[item]

        return None

    def user_rating(self, user, restaurant):
        try:
            return self.ratings_of_user[user][restaurant]
        except KeyError:
            return None

    def restaurant_rating(self, restaurant, user):
        try:
            return self.ratings_of_restaurant[restaurant][user]
        except KeyError:
            return None

    def aspect_weight(self, item, aspect):
        if item in self.users():
            if aspect in self.user_aspects[item]:
                return self.user_aspects[item][aspect]
        elif item in self.restaurants():
            if aspect in self.restaurant_aspects[item]:
                return self.restaurant_aspects[item][aspect]

        return None  # TODO: create rating_not_exists exception and throw here

    def aspects(self, item):
        if item in self.user_aspects:
            return self.user_aspects[item]
        elif item in self.restaurant_aspects:
            return self.restaurant_aspects[item]

        return None

    def users(self):
        return self.all_users

    def restaurants(self):
        return self.all_restaurants
