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
    """
    Ecuacion (5) del paper
    """

    def __init__(self, ratings):
        super().__init__(ratings)

    def score(self, user, restaurant):
        return abs(self.scalar_product(user, restaurant) / (self.module(user) * self.module(restaurant)))

    def scalar_product(self, user, restaurant):
        aspects_user = set(self.ratings.aspects(user).keys())
        aspects_restaurant = set(self.ratings.aspects(restaurant).keys())
        common_aspects = aspects_user.intersection(aspects_restaurant)

        summation = 0
        for aspect in common_aspects:
            summation += self.ratings.aspect_weight(user, aspect) * self.ratings.aspect_weight(restaurant, aspect)

        return summation

    def module(self, item):
        return math.sqrt(sum(i**2 for i in self.ratings.aspects(item).values()))


class UserKNNRecommender(Recommender):
    """
    Ecuacion (8) del paper
    """

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
        heap = self.knn_similarity[user]

        summation = 0
        for sim, user2 in heap:
            if restaurant in self.ratings.user_ratings(user2).keys():
                if self.ratings.user_rating(user2, restaurant) != 0:
                    summation += sim * self.ratings.user_rating(user2, restaurant)

        return summation


class RestaurantKNNRecommender(Recommender):
    """
    Ecuacion (6) del paper
    """

    def __init__(self, ratings, sim, k):
        super().__init__(ratings)

        self.sim = sim
        self.k = k
        self.knn_similarity = {}

        for restaurant1 in self.ratings.restaurants():
            self.knn_similarity[restaurant1] = []

            for restaurant2 in self.ratings.restaurants():
                if restaurant2 != restaurant1:
                    sim = self.sim.s[restaurant1][restaurant2]

                    if sim > 0:
                        if len(self.knn_similarity[restaurant1]) < self.k:
                            heapq.heappush(self.knn_similarity[restaurant1], (sim, restaurant2))
                        elif sim > self.knn_similarity[restaurant1][0][0]:
                            heapq.heapreplace(self.knn_similarity[restaurant1], (sim, restaurant2))

    def recommend(self, topn):
        """
        :param topn: it limits the Ranking size to topn items.
        :return: a dictionary of a ranking per user.
        """
        recommendations = {}

        for restaurant in self.ratings.restaurants():
            ranking = Ranking(topn)

            for user in self.ratings.users():
                ranking.add(user, self.score(restaurant, user))

            recommendations[restaurant] = ranking

        return recommendations

    def score(self, restaurant, user):
        heap = self.knn_similarity[restaurant]

        summation = 0
        for sim, restaurant2 in heap:
            if user in self.ratings.restaurant_ratings(restaurant2).keys():
                if self.ratings.restaurant_rating(restaurant2, user) != 0:
                    summation += sim * self.ratings.restaurant_rating(restaurant2, user)

        return summation