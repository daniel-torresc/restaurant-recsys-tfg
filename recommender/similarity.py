import math
from abc import ABC, abstractmethod


class Similarity(ABC):
    def __init__(self, ratings):
        self.ratings = ratings

    @abstractmethod
    def sim(self, item1, item2):
        """ Computation of user-user similarity metric """


class CosineUserSimilarity(Similarity):
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
        aspects_user1 = set(self.ratings.user_aspects(user1).keys())
        aspects_user2 = set(self.ratings.user_aspects(user2).keys())
        common_aspects = aspects_user1.intersection(aspects_user2)

        summation = 0
        for aspect in common_aspects:
            summation += self.ratings.rating(user1, aspect) * self.ratings.rating(user2, aspect)

        return summation

    def module(self, user):
        return math.sqrt(sum(i**2 for i in self.ratings.user_aspects(user).values()))
