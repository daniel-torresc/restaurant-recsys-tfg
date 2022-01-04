import statistics
from abc import ABC, abstractmethod


class Metric(ABC):

    def __init__(self, test, cutoff, threshold):
        self.test = test
        self.cutoff = cutoff
        self.threshold = threshold

    def __repr__(self):
        return f"{type(self).__name__}" + f"@{self.cutoff}" if self.cutoff else ""

    @abstractmethod
    def compute(self, recommendation):
        """ Core computing function of the evaluation metrics """

    def get_relevant_recommendations(self, item, recommendations):
        relevant_recommendations = 0  # A recommendation is relevant when the rating set by the user >= threshold

        for index, rec in enumerate(recommendations, start=1):
            try:
                if item in self.test.users():
                    if self.test.user_rating(item, rec[0]) >= self.threshold:
                        relevant_recommendations += 1
                elif item in self.test.restaurants():
                    if self.test.restaurant_rating(item, rec[0]) >= self.threshold:
                        relevant_recommendations += 1
            except ValueError:
                continue

            if index == self.cutoff:
                break

        return relevant_recommendations


class Precision(Metric):

    def __init__(self, test, cutoff, threshold):
        super().__init__(test, cutoff, threshold)

    def compute(self, recommendation):
        precision_list = []
        for item in recommendation:
            relevant_recommendations = self.get_relevant_recommendations(item, recommendation[item])
            precision = relevant_recommendations / min(self.cutoff, len(recommendation[item]))

            precision_list.append(precision)

        return statistics.mean(precision_list)  # average of all precision metrics


class Recall(Metric):
    def __init__(self, test, cutoff, threshold):
        super().__init__(test, cutoff. cutoff, threshold)

        self.relevant_user = {}
        for user in self.test.users():
            self.relevant_user[user] = 0

            for rating in self.test.ratings(user):
                if rating >= self.threshold:
                    self.relevant_user[user] += 1

        self.relevant_restaurant = {}
        for restaurant in self.test.restaurants():
            self.relevant_restaurant[restaurant] = 0

            for rating in self.test.ratings(restaurant):
                if rating >= self.threshold:
                    self.relevant_restaurant[restaurant] += 1

    def compute(self, recommendation):
        recall_list = []
        for item in recommendation:
            try:
                relevant_recommendations = self.get_relevant_recommendations(item, recommendation[item])
                recall = relevant_recommendations / self.get_num_relevants(item)
            except ZeroDivisionError:
                recall = 0

            recall_list.append(recall)

        return statistics.mean(recall_list)  # average of all recall metrics

    def get_num_relevants(self, item):
        if item in self.test.users():
            return self.relevant_user[item]
        elif item in self.test.restaurants():
            return self.relevant_restaurant[item]
        else:
            return 0
