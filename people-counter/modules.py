class PeopleCounter:
    def __init__(self):
        self.counts = []

    def parse(self, results):
        self.counts = self.counts[-100:] + [len(results)]
        return max(set(self.counts), key=self.counts.count)
