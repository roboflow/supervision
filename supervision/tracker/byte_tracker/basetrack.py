class BaseTrack:
    def __init__(self):
        self.start_frame = 0
        self.frame_id = 0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    def reset_counter(self):
        self.frame_id = 0
        self.start_frame = 0

    def activate(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError
