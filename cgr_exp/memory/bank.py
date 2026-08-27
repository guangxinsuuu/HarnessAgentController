class MemoryBank:
    def __init__(self):
        self.frozen = False
        self.episodes = []

    def add(self, episode):
        if self.frozen:
            raise RuntimeError("memory bank is frozen")
        self.episodes.append(episode)

    def freeze(self):
        self.frozen = True
