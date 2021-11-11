import math
from abc import ABC, abstractmethod


class Similarity(ABC):
    def __init__(self, ratings):
        self.ratings = ratings

    @abstractmethod
    def sim(self, item1, item2):
        """ Computation of user-user similarity metric """


class CosineUserSimilarityAspects(Similarity):

    def __init__(self, ratings):
        super().__init__(ratings)

        self.s = {}  # s[u1][u2] = similarity between users u1 and u2
        for user1 in self.ratings.users():
            self.s[user1] = {}
            for user2 in self.ratings.users():
                self.s[user1][user2] = self.sim(user1, user2)

    def sim(self, user1, user2):
        return abs(self.scalar_product(user1, user2) / (self.module(user1) * self.module(user2)))

    def scalar_product(self, user1, user2):
        aspects_user1 = set(self.ratings.aspects(user1).keys())
        aspects_user2 = set(self.ratings.aspects(user2).keys())
        common_aspects = aspects_user1.intersection(aspects_user2)

        summation = 0
        for aspect in common_aspects:
            summation += self.ratings.aspect_weight(user1, aspect) * self.ratings.aspect_weight(user2, aspect)

        return summation

    def module(self, user):
        return math.sqrt(sum(i**2 for i in self.ratings.aspects(user).values()))


class CosineRestaurantSimilarityAspects(Similarity):

    def __init__(self, ratings):
        super().__init__(ratings)

        self.s = {}  # s[r1][r2] = similarity between restaurants r1 and r2
        for restaurant1 in self.ratings.restaurants():
            self.s[restaurant1] = {}
            for restaurant2 in self.ratings.restaurants():
                self.s[restaurant1][restaurant2] = self.sim(restaurant1, restaurant2)

    def sim(self, restaurant1, restaurant2):
        return abs(self.scalar_product(restaurant1, restaurant2) / (self.module(restaurant1) * self.module(restaurant2)))

    def scalar_product(self, restaurant1, restaurant2):
        aspects_restaurant1 = set(self.ratings.aspects(restaurant1).keys())
        aspects_restaurant2 = set(self.ratings.aspects(restaurant2).keys())
        common_aspects = aspects_restaurant1.intersection(aspects_restaurant2)

        summation = 0
        for aspect in common_aspects:
            summation += self.ratings.aspect_weight(restaurant1, aspect) * self.ratings.aspect_weight(restaurant2, aspect)

        return summation

    def module(self, restaurant):
        return math.sqrt(sum(i**2 for i in self.ratings.aspects(restaurant).values()))


class CosineUserSimilarityRatings(Similarity):

    def __init__(self, ratings):
        super().__init__(ratings)

        self.s = {}  # s[u1][u2] = similarity between users u1 and u2
        for user1 in self.ratings.users():
            self.s[user1] = {}
            for user2 in self.ratings.users():
                self.s[user1][user2] = self.sim(user1, user2)

    def sim(self, user1, user2):
        return abs(self.scalar_product(user1, user2) / (self.module(user1) * self.module(user2)))

    def scalar_product(self, user1, user2):
        restaurants_user1 = set(self.ratings.ratings(user1).keys())
        restaurants_user2 = set(self.ratings.ratings(user2).keys())
        common_restaurants = restaurants_user1.intersection(restaurants_user2)

        summation = 0
        for restaurant in common_restaurants:
            summation += self.ratings.user_rating(user1, restaurant) * self.ratings.user_rating(user2, restaurant)

        return summation

    def module(self, user):
        return math.sqrt(sum(i**2 for i in self.ratings.ratings(user).values()))


class CosineRestaurantSimilarityRatings(Similarity):

    def __init__(self, ratings):
        super().__init__(ratings)

        self.s = {}  # s[r1][r2] = similarity between restaurants r1 and r2
        for restaurant1 in self.ratings.restaurants():
            self.s[restaurant1] = {}
            for restaurant2 in self.ratings.restaurants():
                self.s[restaurant1][restaurant2] = self.sim(restaurant1, restaurant2)

    def sim(self, restaurant1, restaurant2):
        return abs(self.scalar_product(restaurant1, restaurant2) / (self.module(restaurant1) * self.module(restaurant2)))

    def scalar_product(self, restaurant1, restaurant2):
        users_restaurant1 = set(self.ratings.ratings(restaurant1).keys())
        users_restaurant2 = set(self.ratings.ratings(restaurant2).keys())
        common_users = users_restaurant1.intersection(users_restaurant2)

        summation = 0
        for user in common_users:
            summation += self.ratings.restaurant_rating(restaurant1, user) * self.ratings.restaurant_rating(restaurant2, user)

        return summation

    def module(self, restaurant):
        return math.sqrt(sum(i**2 for i in self.ratings.ratings(restaurant).values()))
