import collections


class CacheSimulator:
    """Simulate a limited-capacity block cache with LRU or FIFO policy."""

    def __init__(self, capacity_blocks: int, policy: str = "lru"):
        self.capacity = max(0, capacity_blocks)
        self.policy = policy.lower()
        self.cache: dict = {}
        self.order = collections.OrderedDict()
        self.total_evictions = 0
        self.seen = set()

    def _evict_one(self):
        if not self.order:
            return
        oldest_hid, _ = self.order.popitem(last=False)
        self.cache.pop(oldest_hid, None)
        self.total_evictions += 1

    def _insert(self, hid: int):
        if hid in self.cache:
            if self.policy == "lru":
                self.order.move_to_end(hid)
            return
        while len(self.cache) >= self.capacity:
            self._evict_one()
        self.cache[hid] = True
        self.order[hid] = True

    def access_blocks(self, block_ids: list) -> tuple:
        """
        Simulate prefix caching for a single request.

        Returns:
            (hit_blocks, miss_blocks, cold_miss, capacity_miss, evictions_during_this_request)
        """
        hit = 0
        miss = 0
        cold_miss = 0
        capacity_miss = 0
        evictions_before = self.total_evictions
        prefix_break = False

        for hid in block_ids:
            if not prefix_break and hid in self.cache:
                hit += 1
                if self.policy == "lru":
                    self.order.move_to_end(hid)
            else:
                prefix_break = True
                miss += 1
                if hid not in self.seen:
                    cold_miss += 1
                    self.seen.add(hid)
                else:
                    capacity_miss += 1
                self._insert(hid)

        evictions = self.total_evictions - evictions_before
        return hit, miss, cold_miss, capacity_miss, evictions
