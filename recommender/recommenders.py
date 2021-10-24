import heapq
import math
from abc import ABC, abstractmethod
from ranking import Ranking


class Recommender(ABC):

    def __init__(self, ratings):
        self.ratings = ratings

    def __repr__(self):
        return type(self).__name__

    def recommend(self, topn):
        """
        :param topn: it limits the Ranking size to topn items.
        :return: a dictionary of a ranking per user.
        """
        recommendations = {}

        for user in self.ratings.users():
            ranking = Ranking(topn)

            for restaurant in self.ratings.restaurants():
                ranking.add(restaurant, self.score(user, restaurant))

            recommendations[user] = ranking

        return recommendations

    @abstractmethod
    def score(self, user, restaurant):
        """ Core scoring function of the recommendation algorithm """


class CosineRecommender(Recommender):

    def __init__(self, ratings):
        super().__init__(ratings)

    def score(self, user, restaurant):
        return abs(self.scalar_product(user, restaurant) / (self.user_module(user) * self.restaurant_module(restaurant)))

    def scalar_product(self, user, restaurant):
        aspects_user = set(self.ratings.user_aspects(user).keys())
        aspects_restaurant = set(self.ratings.restaurant_aspects(restaurant).keys())
        common_aspects = aspects_user.intersection(aspects_restaurant)

        summation = 0
        for aspect in common_aspects:
            summation += self.ratings.rating(user, aspect) * self.ratings.rating(restaurant, aspect)

        return summation

    def user_module(self, user):
        return math.sqrt(sum(i**2 for i in self.ratings.user_aspects(user).values()))

    def restaurant_module(self, restaurant):
        return math.sqrt(sum(i**2 for i in self.ratings.restaurant_aspects(restaurant).values()))


class UserKNNRecommender(Recommender):

    def __init__(self, ratings, sim, k):
        super().__init__(ratings)

        self.sim = sim
        self.k = k
        self.knn_similarity = {}

        for user1 in self.ratings.users():
            self.knn_similarity[user1] = []

            for user2 in self.ratings.users():
                if user2 != user1:
                    sim = self.sim.s[user1][user2]

                    if sim > 0:
                        if len(self.knn_similarity[user1]) < self.k:
                            heapq.heappush(self.knn_similarity[user1], (sim, user2))
                        elif sim > self.knn_similarity[user1][0][0]:
                            heapq.heapreplace(self.knn_similarity[user1], (sim, user2))

    def score(self, user, restaurant):
        pass
        # heap = self.knn_similarity[user]
        #
        # summation = 0
        # for sim, user2 in heap:
        #     if restaurant in self.ratings.user_aspects(user2).keys():
        #         if self.ratings.rating(user2, restaurant) != 0:
        #             summation += sim * self.training.rating(user2, restaurant)
        #
        # return summation
