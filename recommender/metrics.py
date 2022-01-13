import statistics
from abc import ABC, abstractmethod


class Metric(ABC):

    def __init__(self, test, cutoff, threshold):
        self.test = test
        self.cutoff = cutoff
        self.threshold = threshold

    def __repr__(self):
        return f"{type(self).__name__}" + (f"@{self.cutoff}" if self.cutoff else "") + f" threshold={self.threshold}"

    @abstractmethod
    def compute(self, recommendation):
        """ Core computing function of the evaluation metrics """

    def get_relevant_recommendations(self, user, recommendations):
        relevant_recommendations = 0  # A recommendation is relevant when the rating set by the user >= threshold

        for index, rec in enumerate(recommendations, start=1):
            rate = self.test.user_rating(user, rec[0])
            if rate != 0 and rate >= self.threshold:
                relevant_recommendations += 1

            if index == self.cutoff:
                break

        return relevant_recommendations


class Precision(Metric):

    def __init__(self, test, cutoff, threshold):
        super().__init__(test, cutoff, threshold)

    def compute(self, recommendation):
        precision_list = []
        for user in recommendation:
            relevant_recommendations = self.get_relevant_recommendations(user, recommendation[user])
            precision = relevant_recommendations / min(self.cutoff, len(recommendation[user]))

            precision_list.append(precision)

        return statistics.mean(precision_list), statistics.stdev(precision_list)


class Recall(Metric):
    def __init__(self, test, cutoff, threshold):
        super().__init__(test, cutoff, threshold)

        self.relevant_user = {}
        for user in self.test.users():
            self.relevant_user[user] = 0

            for rating in self.test.ratings(user):
                if self.test.ratings(user)[rating] >= self.threshold:
                    self.relevant_user[user] += 1

    def compute(self, recommendation):
        recall_list = []
        for user in recommendation:
            try:
                relevant_recommendations = self.get_relevant_recommendations(user, recommendation[user])
                recall = relevant_recommendations / self.get_num_relevants(user)
            except ZeroDivisionError:
                recall = 0

            recall_list.append(recall)

        return statistics.mean(recall_list), statistics.stdev(recall_list)

    def get_num_relevants(self, obj):
        try:
            return self.relevant_user[obj]
        except KeyError:
            return 0
