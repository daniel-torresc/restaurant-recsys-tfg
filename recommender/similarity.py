import math
from abc import ABC, abstractmethod


class Similarity(ABC):
    def __init__(self, ratings):
        self.ratings = ratings
        
    def __repr__(self):
        return type(self).__name__

    @abstractmethod
    def sim(self, item1, item2):
        """ Computation of user-user similarity metric """


class CosineUserSimilarityAspects(Similarity):

    def __init__(self, ratings):
        print(f"Building {self}...", end='', flush=True)

        super().__init__(ratings)

        # Cache variables
        self.module_cache = {}
        self.aspects_cache = {}

        self.s = {}  # s[u1][u2] = similarity between users u1 and u2
        for index1, user1 in enumerate(self.ratings.users()):
            self.s[user1] = {}

            for index2, user2 in enumerate(self.ratings.users()):
                if index1 < index2:
                    break

                self.s.setdefault(user2, {})  # Create dictionary just in case it doesn't already exist

                # As it is a symmetric matrix, we can set both at once time and save some computation time
                sim = self.sim(user1, user2)
                self.s[user1][user2] = sim
                self.s[user2][user1] = sim

        print("DONE")

    def sim(self, user1, user2):
        return abs(self.scalar_product(user1, user2) / (self.module(user1) * self.module(user2)))

    def scalar_product(self, user1, user2):
        aspects_user1 = self.aspects(user1)
        aspects_user2 = self.aspects(user2)
        common_aspects = aspects_user1.intersection(aspects_user2)

        return sum(
            self.ratings.aspect_weight(user1, aspect)
            * self.ratings.aspect_weight(user2, aspect)
            for aspect in common_aspects
        )

    def module(self, user):
        if user in self.module_cache:
            return self.module_cache[user]
        else:
            self.module_cache[user] = math.sqrt(sum(i ** 2 for i in self.ratings.aspects(user).values()))
            return self.module_cache[user]

    def aspects(self, user):
        if user in self.aspects_cache:
            return self.aspects_cache[user]
        else:
            self.aspects_cache[user] = set(self.ratings.aspects(user).keys())
            return self.aspects_cache[user]


class CosineRestaurantSimilarityAspects(Similarity):

    def __init__(self, ratings):
        print(f"Building {self}...", end='', flush=True)

        super().__init__(ratings)

        # Cache variables
        self.module_cache = {}
        self.aspects_cache = {}

        self.s = {}  # s[r1][r2] = similarity between restaurants r1 and r2
        for index1, restaurant1 in enumerate(self.ratings.restaurants()):
            self.s[restaurant1] = {}

            for index2, restaurant2 in enumerate(self.ratings.restaurants()):
                if index1 < index2:
                    break

                self.s.setdefault(restaurant2, {})  # Create dictionary just in case it doesn't already exist

                # As it is a symmetric matrix, we can set both at once time and save some computation time
                sim = self.sim(restaurant1, restaurant2)
                self.s[restaurant1][restaurant2] = sim
                self.s[restaurant2][restaurant1] = sim

        print("DONE")

    def sim(self, restaurant1, restaurant2):
        return abs(self.scalar_product(restaurant1, restaurant2) / (self.module(restaurant1) * self.module(restaurant2)))

    def scalar_product(self, restaurant1, restaurant2):
        aspects_restaurant1 = self.aspects(restaurant1)
        aspects_restaurant2 = self.aspects(restaurant2)
        common_aspects = aspects_restaurant1.intersection(aspects_restaurant2)

        return sum(
            self.ratings.aspect_weight(restaurant1, aspect)
            * self.ratings.aspect_weight(restaurant2, aspect)
            for aspect in common_aspects
        )

    def module(self, restaurant):
        if restaurant in self.module_cache:
            return self.module_cache[restaurant]
        else:
            self.module_cache[restaurant] = math.sqrt(sum(i ** 2 for i in self.ratings.aspects(restaurant).values()))
            return self.module_cache[restaurant]

    def aspects(self, restaurant):
        if restaurant in self.aspects_cache:
            return self.aspects_cache[restaurant]
        else:
            self.aspects_cache[restaurant] = set(self.ratings.aspects(restaurant).keys())
            return self.aspects_cache[restaurant]


class CosineUserSimilarityRatings(Similarity):

    def __init__(self, ratings):
        print(f"Building {self}...", end='', flush=True)

        super().__init__(ratings)

        # Cache variables
        self.module_cache = {}
        self.restaurants_cache = {}

        self.s = {}  # s[u1][u2] = similarity between users u1 and u2
        for index1, user1 in enumerate(self.ratings.users()):
            self.s[user1] = {}

            for index2, user2 in enumerate(self.ratings.users()):
                if index1 < index2:
                    break

                self.s.setdefault(user2, {})  # Create dictionary just in case it doesn't already exist

                # As it is a symmetric matrix, we can set both at once time and save some computation time
                sim = self.sim(user1, user2)
                self.s[user1][user2] = sim
                self.s[user2][user1] = sim

        print("DONE")

    def sim(self, user1, user2):
        return abs(self.scalar_product(user1, user2) / (self.module(user1) * self.module(user2)))

    def scalar_product(self, user1, user2):
        restaurants_user1 = self.restaurants(user1)
        restaurants_user2 = self.restaurants(user2)
        common_restaurants = restaurants_user1.intersection(restaurants_user2)

        return sum(
            self.ratings.user_rating(user1, restaurant)
            * self.ratings.user_rating(user2, restaurant)
            for restaurant in common_restaurants
        )

    def module(self, user):
        if user in self.module_cache:
            return self.module_cache[user]
        else:
            self.module_cache[user] = math.sqrt(sum(i ** 2 for i in self.ratings.ratings(user).values()))
            return self.module_cache[user]

    def restaurants(self, user):
        if user in self.restaurants_cache:
            return self.restaurants_cache[user]
        else:
            self.restaurants_cache[user] = set(self.ratings.ratings(user).keys())
            return self.restaurants_cache[user]


class CosineRestaurantSimilarityRatings(Similarity):

    def __init__(self, ratings):
        print(f"Building {self}...", end='', flush=True)

        super().__init__(ratings)

        # Cache variables
        self.module_cache = {}
        self.users_cache = {}

        self.s = {}  # s[r1][r2] = similarity between restaurants r1 and r2
        for index1, restaurant1 in enumerate(self.ratings.restaurants()):
            self.s[restaurant1] = {}

            for index2, restaurant2 in enumerate(self.ratings.restaurants()):
                if index1 < index2:
                    break

                self.s.setdefault(restaurant2, {})  # Create dictionary just in case it doesn't already exist

                # As it is a symmetric matrix, we can set both at once time and save some computation time
                sim = self.sim(restaurant1, restaurant2)
                self.s[restaurant1][restaurant2] = sim
                self.s[restaurant2][restaurant1] = sim

        print("DONE")

    def sim(self, restaurant1, restaurant2):
        return abs(self.scalar_product(restaurant1, restaurant2) / (self.module(restaurant1) * self.module(restaurant2)))

    def scalar_product(self, restaurant1, restaurant2):
        users_restaurant1 = self.users(restaurant1)
        users_restaurant2 = self.users(restaurant2)
        common_users = users_restaurant1.intersection(users_restaurant2)

        return sum(
            self.ratings.restaurant_rating(restaurant1, user)
            * self.ratings.restaurant_rating(restaurant2, user)
            for user in common_users
        )

    def module(self, restaurant):
        if restaurant in self.module_cache:
            return self.module_cache[restaurant]
        else:
            self.module_cache[restaurant] = math.sqrt(sum(i ** 2 for i in self.ratings.ratings(restaurant).values()))
            return self.module_cache[restaurant]

    def users(self, restaurant):
        if restaurant in self.users_cache:
            return self.users_cache[restaurant]
        else:
            self.users_cache[restaurant] = set(self.ratings.ratings(restaurant).keys())
            return self.users_cache[restaurant]
