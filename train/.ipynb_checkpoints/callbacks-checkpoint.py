# train/callbacks.py (minimal)
class EarlyStop:
    def __init__(self, patience=3):
        self.best = 1e9; self.count = 0; self.p = patience
    def __call__(self, metric):
        if metric < self.best:
            self.best = metric; self.count = 0
        else:
            self.count += 1
        return self.count >= self.p
