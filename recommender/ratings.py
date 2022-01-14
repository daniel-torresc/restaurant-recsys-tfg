import statistics


class Ratings:

    def __init__(self, df):
        self.records_of_user = {}
        self.records_of_restaurant = {}

        records = df.to_dict(orient='records')
        for record in records:
            dict_aux = {
                'aspect': record['aspect'],
                'rate': record['rate'],
                'restaurant': record['restaurant_id'],
                'feeling': record['feeling']
            }
            self.records_of_user.setdefault(record['user_id'], [])
            self.records_of_user[record['user_id']].append(dict_aux)

            dict_aux = {
                'aspect': record['aspect'],
                'rate': record['rate'],
                'user': record['user_id'],
                'feeling': record['feeling']
            }
            self.records_of_restaurant.setdefault(record['restaurant_id'], [])
            self.records_of_restaurant[record['restaurant_id']].append(dict_aux)

        self.all_users = list(self.records_of_user.keys())
        self.all_restaurants = list(self.records_of_restaurant.keys())

        self.user_aspects = {}  # weighted aspects for each user
        self.restaurant_aspects = {}  # weighted aspects for each restaurant

        self.ratings_of_user = {}  # scores of the user to the restaurants
        self.ratings_of_restaurant = {}  # scores of the restaurant from the users

        # Calculate the weight of each aspect for every user
        for user in self.records_of_user:
            self.user_aspects[user] = {}
            self.ratings_of_user[user] = {}

            for record in self.records_of_user[user]:
                aspect = record['aspect']
                rate = record['rate']
                restaurant = record['restaurant']
                feeling = record['feeling']

                weighted_score = 0.6 * rate + 0.4 * feeling * 5

                self.user_aspects[user].setdefault(aspect, weighted_score)
                self.user_aspects[user][aspect] = statistics.mean([self.user_aspects[user][aspect], weighted_score])

                self.ratings_of_user[user][restaurant] = rate

        # Calculate the weight of each aspect for every restaurant
        for restaurant in self.records_of_restaurant:
            self.restaurant_aspects[restaurant] = {}
            self.ratings_of_restaurant[restaurant] = {}

            for record in self.records_of_restaurant[restaurant]:
                aspect = record['aspect']
                rate = record['rate']
                user = record['user']
                feeling = record['feeling']

                weighted_score = 0.6 * rate + 0.4 * feeling * 5

                self.restaurant_aspects[restaurant].setdefault(aspect, weighted_score)
                self.restaurant_aspects[restaurant][aspect] = statistics.mean([self.restaurant_aspects[restaurant][aspect], weighted_score])

                self.ratings_of_restaurant[restaurant][user] = rate

    def ratings(self, obj):
        if obj in self.ratings_of_user:
            return self.ratings_of_user[obj]
        elif obj in self.ratings_of_restaurant:
            return self.ratings_of_restaurant[obj]

        return None

    def user_rating(self, user, restaurant):
        try:
            return self.ratings_of_user[user][restaurant]
        except KeyError:
            return 0

    def restaurant_rating(self, restaurant, user):
        try:
            return self.ratings_of_restaurant[restaurant][user]
        except KeyError:
            return 0

    def aspect_weight(self, obj, aspect):
        if obj in self.user_aspects:
            if aspect in self.user_aspects[obj]:
                return self.user_aspects[obj][aspect]
        elif obj in self.restaurant_aspects:
            if aspect in self.restaurant_aspects[obj]:
                return self.restaurant_aspects[obj][aspect]

        return 0

    def aspects(self, obj):
        if obj in self.user_aspects:
            return self.user_aspects[obj]
        elif obj in self.restaurant_aspects:
            return self.restaurant_aspects[obj]

        return None

    def users(self):
        return self.all_users

    def restaurants(self):
        return self.all_restaurants
