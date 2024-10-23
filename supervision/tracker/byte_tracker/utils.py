class IdCounter:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._id = self.NO_ID

    def new_id(self) -> int:
        self._id += 1
        return self._id

    @property
    def NO_ID(self) -> int:
        return 0
