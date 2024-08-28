"""
The cache is used to store the segment counts. E.g.
{
    'atg': {
        count: 3,
        indices: {0, 1, 100}
        ttl: 1
    }
}
"""


class Cache:
    def __init__(self):
        self.cache = {}

    def __len__(self):
        return len(self.cache)

    def __contains__(self, segment: str):
        return segment in self.cache

    def get(self, segment: str):
        if segment in self.cache:
            self.cache[segment]['ttl'] += 1

            return {
                'count': self.cache[segment]['count'],
                'indices': list(self.cache[segment]['indices'])
            }

        return None

    def set(self, segment: str, index: int):
        if segment not in self.cache:
            self.cache[segment] = {
                'count': 0,
                'indices': set(),
                'ttl': 0
            }

        self.cache[segment]['indices'].add(index)
        self.cache[segment]['count'] = len(self.cache[segment]['indices'])
        self.cache[segment]['ttl'] = 1

    def clear(self):
        self.cache = {}

    def refresh(self):
        """
        Reduce the ttl for all and remove the segments that have a ttl of less than 0.
        """
        self.cache = {
            segment: {
                'count': self.cache[segment]['count'],
                'indices': self.cache[segment]['indices'],
                'ttl': self.cache[segment]['ttl'] - 1
            }
            for segment in self.cache if self.cache[segment]['ttl'] >= 1
        }
