"""Shared bounded history: window size counts observations, including the current one."""


class ObservationHistory:
    def __init__(self, window_size=1):
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("history_window_size must be a positive integer")
        self.window_size = window_size
        self.turns = []

    def append(self, message, images=()):
        self.turns.append((message, list(images)))
        # Drop complete oldest user/assistant pairs, never an isolated image.
        while sum(m.get("role") == "user" for m, _ in self.turns) > self.window_size:
            self.turns.pop(0)
            while self.turns and self.turns[0][0].get("role") != "user":
                self.turns.pop(0)

    @property
    def messages(self):
        return [m for m, _ in self.turns]

    def drop_oldest_turn(self):
        if sum(m.get("role") == "user" for m, _ in self.turns) <= 1:
            return False
        self.turns.pop(0)
        while self.turns and self.turns[0][0].get("role") != "user":
            self.turns.pop(0)
        return True

    @property
    def images(self):
        return [image for _, images in self.turns for image in images]
