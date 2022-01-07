import heapq


class Ranking:
    class ScoredItem:
        """
        Clase utilizada para gestionar las comparaciones que se realizan dentro del heap
        """
        def __init__(self, element):
            self.element = element

        def __lt__(self, other):
            """
            En primer lugar se compara el score. En caso de que sean iguales (mismo score),
            se compara usando el itemid (se colocará más arriba el elemento con un itemid menor).
            """
            return self.element[0] < other.element[0] if self.element[0] != other.element[0] \
                else self.element[1] > other.element[1]

        def __eq__(self, other):
            return self.element == other.element

        def __str__(self):
            return str(self.element)

        def __repr__(self):
            return self.__str__()

    def __init__(self, topn):
        self.heap = []
        self.topn = topn
        self.changed = 0

    def add(self, item, score):
        scored_item = self.ScoredItem((score, item))
        if len(self.heap) < self.topn:
            heapq.heappush(self.heap, scored_item)
            self.changed = 1
        elif scored_item > self.heap[0]:
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, scored_item)
            self.changed = 1

    def __iter__(self):
        if self.changed:
            self.ranking = []
            h = self.heap.copy()
            while h:
                self.ranking.append(heapq.heappop(h).element[::-1])
            self.changed = 0
        return reversed(self.ranking)

    def __repr__(self):
        r = ""
        for item, score in self:
            r += f"\t\tRestaurant [{item}] : {score}\n"

        return r[:-1]

    def __len__(self):
        return len([x for x in self])
