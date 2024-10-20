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
    def __init__(self, num_nodes: int):
        self._caches = [{} for _ in range(num_nodes)]

    def __len__(self):
        return len(self._caches)

    def __contains__(self, segment: str):
        return segment in self._caches

    def get(self, segment: str, node_id: int):
        if segment in self._caches[node_id]:
            self._caches[node_id][segment]['ttl'] += 1

            return {
                'count': self._caches[node_id][segment]['count'],
                'indices': list(self._caches[node_id][segment]['indices'])
            }

        return None

    def set(self, segment: str, index: int, node_id: int):
        if segment not in self._caches[node_id]:
            self._caches[node_id][segment] = {
                'count': 0,
                'indices': set(),
                'ttl': 0
            }

        self._caches[node_id][segment]['indices'].add(index)
        self._caches[node_id][segment]['count'] = len(self._caches[node_id][segment]['indices'])
        self._caches[node_id][segment]['ttl'] = 1

    def clear(self):
        self._caches = {}

    def refresh(self):
        """
        Reduce the ttl for all and remove the segments that have a ttl of less than 0.
        """
        self._caches = [{
            segment: {
                'count': cache[segment]['count'],
                'indices': cache[segment]['indices'],
                'ttl': cache[segment]['ttl'] - 1
            }
            for segment in cache if cache[segment]['ttl'] >= 1
        } for cache in self._caches]
