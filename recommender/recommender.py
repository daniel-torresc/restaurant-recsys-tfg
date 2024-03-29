import heapq
import math
from abc import ABC, abstractmethod
from ranking import Ranking


class Recommender(ABC):

    def __init__(self, ratings):
        self.ratings = ratings

    def __repr__(self):
        return type(self).__name__

    def recommend(self, topn, test_users=None):
        """
        :param topn: it limits the Ranking size to topn items.
        :param test_users: users to test
        :return: a dictionary of a ranking per user.
        """
        recommendations = {}

        if test_users is None:
            test_users = self.ratings.users()

        for index, user in enumerate(test_users):
            ranking = Ranking(topn)

            for restaurant in self.ratings.restaurants():
                if self.ratings.user_rating(user, restaurant) == 0:
                    ranking.add(restaurant, self.score(user, restaurant))

            recommendations[user] = ranking

            if (index+1) % 100 == 0:
                print(f"\t\tRecommmended {index+1} out of {len(test_users)} users")

        return recommendations

    @abstractmethod
    def score(self, user, restaurant):
        """ Core scoring function of the recommendation algorithm """


class CosineRecommender(Recommender):
    """
    Ecuacion (5) del paper
    """

    def __init__(self, ratings):
        print(f"Building {self}...", end='', flush=True)

        super().__init__(ratings)

        # Cache variables
        self.module_cache = {}

        print("DONE")

    def score(self, user, restaurant):
        return abs(self.scalar_product(user, restaurant) / (self.module(user) * self.module(restaurant)))

    def scalar_product(self, user, restaurant):
        aspects_user = set(self.ratings.aspects(user).keys())
        aspects_restaurant = set(self.ratings.aspects(restaurant).keys())
        common_aspects = aspects_user.intersection(aspects_restaurant)

        return sum(
            self.ratings.aspect_user_weight(user, aspect)
            * self.ratings.aspect_restaurant_weight(restaurant, aspect)
            for aspect in common_aspects
        )

    def module(self, item):
        if item not in self.module_cache:
            self.module_cache[item] = math.sqrt(sum(i ** 2 for i in self.ratings.aspects(item).values()))
        return self.module_cache[item]


class UserKNNRecommender(Recommender):
    """
    Ecuacion (8) del paper
    """

    def __init__(self, ratings, sim, k):
        print(f"Building {self}...", end='', flush=True)

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

        print("DONE")

    def score(self, user, restaurant):
        heap = self.knn_similarity[user]

        return sum(
            sim * self.ratings.user_rating(user2, restaurant)
            for sim, user2 in heap
        )


class RestaurantKNNRecommender(Recommender):
    """
    Ecuacion (6) del paper
    """

    def __init__(self, ratings, sim, k):
        print(f"Building {self}...", end='', flush=True)

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

        print("DONE")

    def recommend(self, topn, test_users=None):
        """
        :param topn: it limits the Ranking size to topn items.
        :param test_users: restaurants to test
        :return: a dictionary of a ranking per user.
        """
        recommendations = {}

        if test_users is None:
            test_users = self.ratings.users()

        for index, user in enumerate(test_users):
            ranking = Ranking(topn)

            for restaurant in self.ratings.restaurants():
                if self.ratings.user_rating(user, restaurant) == 0:
                    ranking.add(restaurant, self.score(restaurant, user))

            recommendations[user] = ranking

            if (index+1) % 100 == 0:
                print(f"\t\tRecommmended {index+1} out of {len(test_users)} users")

        return recommendations

    def score(self, restaurant, user):
        heap = self.knn_similarity[restaurant]

        return sum(
            sim * self.ratings.restaurant_rating(restaurant2, user)
            for sim, restaurant2 in heap
        )
