import time
class SimpleRateLimiter:
    def __init__(self, qps: float = 5.0):
        self.min_interval = 1.0 / qps
        self.last = 0.0
    def wait(self):
        now = time.time()
        sleep = max(0.0, self.min_interval - (now - self.last))
        if sleep: time.sleep(sleep)
        self.last = time.time()
