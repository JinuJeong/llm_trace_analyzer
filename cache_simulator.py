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
        self.last_evicted = {}  # hid -> (req_idx, timestamp)
        self.reaccess_gaps = []  # list of (hid, gap_reqs, gap_ts)

    def _evict_one(self, req_idx, timestamp):
        if not self.order:
            return
        oldest_hid, _ = self.order.popitem(last=False)
        self.cache.pop(oldest_hid, None)
        self.total_evictions += 1
        self.last_evicted[oldest_hid] = (req_idx, timestamp)

    def _insert(self, hid, req_idx, timestamp):
        if hid in self.cache:
            if self.policy == "lru":
                self.order.move_to_end(hid)
            return
        while len(self.cache) >= self.capacity:
            self._evict_one(req_idx, timestamp)
        self.cache[hid] = True
        self.order[hid] = True

    def access_blocks(self, block_ids, req_idx=0, timestamp=0):
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
                    if hid in self.last_evicted:
                        ev_idx, ev_ts = self.last_evicted[hid]
                        self.reaccess_gaps.append((
                            hid,
                            req_idx - ev_idx,
                            timestamp - ev_ts,
                        ))
                self._insert(hid, req_idx, timestamp)

        evictions = self.total_evictions - evictions_before
        return hit, miss, cold_miss, capacity_miss, evictions
